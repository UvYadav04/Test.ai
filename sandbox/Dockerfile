# Sandbox image for executing model-generated pandas/DuckDB code in isolation.
# No network access at runtime (container is started with networking disabled by the host - see
# sandbox_manager.py's network_disabled=True), and this image only ever contains the fixed
# sandbox_server.py + execution_engine.py + path_resolver.py - never any model-generated code.
# path_resolver.py is pure stdlib (os/re/uuid) so bundling it here adds nothing to the
# dependency/attack surface - it's the same file host-side code imports as
# `sandbox.path_resolver`, kept as one canonical copy instead of two that could drift.
#
# One persistent container now serves an entire investigation's worth of run_python() calls
# (see sandbox_manager.py/sandbox_server.py) instead of the old one-shot-per-call runner.py
# entrypoint (see git history) - fastapi/uvicorn are what let this be a long-lived HTTP-over-
# Unix-Domain-Socket server instead of a script that exits after one exec().
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    pandas==2.2.2 pyarrow==16.1.0 duckdb==1.0.0 \
    fastapi==0.115.6 "uvicorn[standard]==0.32.1"

COPY sandbox_server.py execution_engine.py path_resolver.py /app/
WORKDIR /app

ENV PYTHONUNBUFFERED=1

# sandbox_server.py itself removes any stale socket file left behind by a previous run before
# binding a fresh one (see its _cleanup_stale_socket) - nothing extra needed at the shell level
# here for that.
ENTRYPOINT ["python", "sandbox_server.py"]
