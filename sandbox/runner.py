"""SUPERSEDED - kept only as a pointer for anything still linking to this filename.

This used to be the sandbox image's ENTRYPOINT: one fresh Docker container per run_python()
call, running this script once, then exiting. It's been replaced by a persistent-sandbox
architecture - one long-lived container per investigation, serving many run_python() calls over
a warm process instead of paying container-create + Python-interpreter-boot + `import
pandas/duckdb` on every single call. See:

- execution_engine.py   - this file's exec()/dfs/describe/preview/sql/save logic, moved
                           near-verbatim into ExecutionEngine, now callable many times per
                           process instead of once-then-exit.
- sandbox_server.py      - the new ENTRYPOINT (see Dockerfile): a FastAPI/Uvicorn server bound
                           to a Unix Domain Socket, exposing ExecutionEngine over
                           GET /health, POST /execute, POST /reset, POST /shutdown.
- sandbox_manager.py     - host-side: creates/reuses one sandbox container per
                           investigation_id, waits for /health, and cleans up on idle timeout
                           or when the investigation ends.
- ../tools/tabular/sandbox_executor.py - PythonSandbox, whose public run(code, tables,
                           workspace_id) API is unchanged; it now talks to sandbox_manager.py
                           instead of spinning up a container itself.

Nothing imports this module - it isn't COPYd into the sandbox image anymore (see Dockerfile).
"""
