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
SOCKET_MOUNT = os.environ.get("SANDBOX_SOCKET_ROOT", "/shared")
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
    logger.info(
        "sandbox server boot: sandbox_id=%s SANDBOX_SOCKET_ROOT env=%s (resolved SOCKET_MOUNT=%s)",
        SANDBOX_ID, os.environ.get("SANDBOX_SOCKET_ROOT"), SOCKET_MOUNT,
    )
    mount_existed_already = os.path.isdir(SOCKET_MOUNT)
    os.makedirs(SOCKET_MOUNT, exist_ok=True)
    if not mount_existed_already:
        logger.warning(
            "sandbox server boot: SOCKET_MOUNT %s did NOT already exist - if this was "
            "supposed to be a bind-mounted volume, the mount is missing/misconfigured and "
            "the socket this server writes here will NOT be visible on the host",
            SOCKET_MOUNT,
        )
    try:
        existing_entries = os.listdir(SOCKET_MOUNT)
    except OSError as exc:
        existing_entries = [f"<could not list: {exc}>"]
    logger.info(
        "sandbox server boot: SOCKET_MOUNT %s contents before cleanup: %s",
        SOCKET_MOUNT, existing_entries,
    )

    _cleanup_stale_socket()
    logger.info("sandbox server starting: sandbox_id=%s binding UDS socket at %s", SANDBOX_ID, SOCKET_PATH)
    try:
        uvicorn.run(app, uds=SOCKET_PATH, log_level="info")
    finally:
        logger.info("sandbox server stopped: sandbox_id=%s - cleaning up socket file", SANDBOX_ID)
        if os.path.exists(SOCKET_PATH):
            try:
                os.remove(SOCKET_PATH)
                logger.info("sandbox server shutdown: removed socket file %s", SOCKET_PATH)
            except OSError:
                logger.exception("failed to remove socket file %s on shutdown", SOCKET_PATH)


if __name__ == "__main__":
    main()
