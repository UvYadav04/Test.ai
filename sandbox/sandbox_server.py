"""Entrypoint for the sandbox Docker image (see Dockerfile) - replaces the old one-shot
runner.py. Starts a long-lived FastAPI/Uvicorn server bound to a Unix Domain Socket (never a TCP
port - this container has no network access at all, see network_disabled=True in
sandbox_manager.py) and keeps it running for the lifetime of one investigation, serving every
run_python() call the host makes during that investigation over the same warm process.

The socket lives on a named Docker volume shared with the worker_service host process (see
sandbox_manager.py's SANDBOX_SOCKET_VOLUME_NAME / SANDBOX_SOCKET_CONTAINER_MOUNT and
docker-compose.yml's `sandbox_sockets` volume) - that's the ONLY way these two otherwise-isolated
sibling containers can see the same .sock file, the exact same named-volume trick
sandbox_manager.py/path_resolver.py already use for the parquet data itself (see those files'
own docstrings for why a named volume, not a bind mount, is what makes docker-outside-of-docker
work here at all).

SANDBOX_ID (an env var set by SandboxManager to the investigation_id) determines this
container's own socket filename - flat inside the shared volume root (no per-investigation
subdirectory: standalone `docker run`/docker-py volume mounts can't bind a sub-path of a named
volume, only the whole volume, so every sandbox container mounts the SAME volume root and picks
its own file within it by name instead)."""
import logging
import os
import signal
import threading
import time
import traceback

import uvicorn
from execution_engine import ExecutionEngine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sandbox.server")

SANDBOX_ID = os.environ.get("SANDBOX_ID", "default")
SOCKET_MOUNT = "/shared"
SOCKET_PATH = os.path.join(SOCKET_MOUNT, f"{SANDBOX_ID}.sock")

app = FastAPI(title="dataanalyzer-sandbox", docs_url=None, redoc_url=None)
engine = ExecutionEngine()
STARTED_AT = time.time()


class ExecuteRequest(BaseModel):
    code: str
    tables: dict[str, str]
    workspace_id: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sandbox_id": SANDBOX_ID,
        "uptime_s": round(time.time() - STARTED_AT, 1),
        "executions": engine.executions,
    }


@app.post("/execute")
def execute(req: ExecuteRequest):
    t0 = time.perf_counter()
    logger.info(
        "request start: POST /execute (sandbox=%s, workspace=%s, tables=%d)",
        SANDBOX_ID, req.workspace_id, len(req.tables),
    )
    try:
        result = engine.execute(req.code, req.tables, req.workspace_id)
    except Exception:
        tb = traceback.format_exc()
        logger.exception("request failed: POST /execute (sandbox=%s)", SANDBOX_ID)
        raise HTTPException(status_code=500, detail=tb[-2000:]) from None
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("request end: POST /execute (sandbox=%s) completed in %.1fms", SANDBOX_ID, elapsed_ms)
    return result


@app.post("/reset")
def reset():
    logger.info("request: POST /reset (sandbox=%s)", SANDBOX_ID)
    engine.reset()
    return {"status": "reset", "sandbox_id": SANDBOX_ID}


@app.post("/shutdown")
def shutdown():
    """Acknowledges the request, then triggers a graceful process exit shortly after responding
    (can't stop the server from inside the same request handler that's serving the response for
    it) - SandboxManager also just removes/kills the container directly on release(), so this
    endpoint is a courtesy for a clean self-shutdown path, not the only way this container dies."""
    logger.info("shutdown requested (sandbox=%s)", SANDBOX_ID)

    def _stop():
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_stop, daemon=True).start()
    return {"status": "shutting down", "sandbox_id": SANDBOX_ID}


def _cleanup_stale_socket() -> None:
    if os.path.exists(SOCKET_PATH):
        logger.warning("removing stale socket file %s before starting", SOCKET_PATH)
        try:
            os.remove(SOCKET_PATH)
        except OSError:
            logger.exception("failed to remove stale socket file %s", SOCKET_PATH)


def main() -> None:
    os.makedirs(SOCKET_MOUNT, exist_ok=True)
    _cleanup_stale_socket()
    logger.info("sandbox server starting: sandbox_id=%s socket=%s", SANDBOX_ID, SOCKET_PATH)
    try:
        uvicorn.run(app, uds=SOCKET_PATH, log_level="info")
    finally:
        logger.info("sandbox server stopped: sandbox_id=%s - cleaning up socket file", SANDBOX_ID)
        if os.path.exists(SOCKET_PATH):
            try:
                os.remove(SOCKET_PATH)
            except OSError:
                logger.exception("failed to remove socket file %s on shutdown", SOCKET_PATH)


if __name__ == "__main__":
    main()
