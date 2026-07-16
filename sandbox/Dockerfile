# Sandbox image for executing model-generated pandas/DuckDB code in isolation.
# No network access at runtime (container is started with networking disabled by the host),
# and this image only ever contains the fixed runner.py - never any model-generated code.
FROM python:3.11-slim

RUN pip install --no-cache-dir pandas==2.2.2 pyarrow==16.1.0 duckdb==1.0.0

COPY runner.py /app/runner.py
WORKDIR /app

ENTRYPOINT ["python", "runner.py"]
