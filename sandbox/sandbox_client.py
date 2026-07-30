import logging
import time

import requests
import requests_unixsocket

logger = logging.getLogger("sandbox.client")


class SandboxClientError(RuntimeError):
    pass


def _quote_socket_path(socket_path: str) -> str:
    return requests.compat.quote(socket_path, safe="")


class SandboxClient:
    def     __init__(self, socket_path: str, timeout_seconds: float = 30.0):
        self.socket_path = socket_path
        self.timeout_seconds = timeout_seconds
        self._session = requests_unixsocket.Session()
        self._base_url = f"http+unix://{_quote_socket_path(socket_path)}"

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def health(self, timeout: float = 2.0) -> dict:
        logger.debug("UDS connection: GET /health via %s", self.socket_path)
        try:
            resp = self._session.get(self._url("/health"), timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise SandboxClientError(f"health check failed for {self.socket_path}: {exc}") from exc

    def execute(self, code: str, tables: dict, workspace_id: str, timeout_seconds: float = None) -> dict:
        timeout = timeout_seconds or self.timeout_seconds
        t0 = time.perf_counter()
        logger.info(
            "UDS request start: POST /execute via %s (workspace=%s, tables=%d, timeout=%.0fs)",
            self.socket_path, workspace_id, len(tables), timeout,
        )
        try:
            resp = self._session.post(
                self._url("/execute"),
                json={"code": code, "tables": tables, "workspace_id": workspace_id},
                timeout=timeout,
            )
        except requests.exceptions.Timeout as exc:
            elapsed = time.perf_counter() - t0
            logger.warning("UDS request TIMED OUT after %.1fs: POST /execute via %s", elapsed, self.socket_path)
            raise SandboxClientError(f"sandbox execution timed out after {timeout}s") from exc
        except Exception as exc:
            logger.error("UDS request failed: POST /execute via %s: %s", self.socket_path, exc)
            raise SandboxClientError(f"could not reach sandbox over UDS {self.socket_path}: {exc}") from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "UDS request end: POST /execute via %s completed in %.1fms (status=%s)",
            self.socket_path, elapsed_ms, resp.status_code,
        )
        if resp.status_code != 200:
            body = resp.text[:500]
            return {
                "stdout": "",
                "saved": [],
                "error": f"sandbox server error ({resp.status_code}): {body}",
            }
        return resp.json()

    def reset(self, timeout: float = 5.0) -> None:
        logger.info("UDS request: POST /reset via %s", self.socket_path)
        try:
            resp = self._session.post(self._url("/reset"), timeout=timeout)
            resp.raise_for_status()
        except Exception as exc:
            raise SandboxClientError(f"reset failed for {self.socket_path}: {exc}") from exc

    def shutdown(self, timeout: float = 5.0) -> None:
        logger.info("UDS request: POST /shutdown via %s", self.socket_path)
        try:
            self._session.post(self._url("/shutdown"), timeout=timeout)
        except Exception:
            logger.debug("no clean response from /shutdown via %s (expected)", self.socket_path)

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass
