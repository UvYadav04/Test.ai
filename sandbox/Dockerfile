FROM python:3.11-slim

RUN pip install --no-cache-dir \
    pandas==2.2.2 pyarrow==16.1.0 duckdb==1.0.0 \
    fastapi==0.115.6 "uvicorn[standard]==0.32.1"

COPY sandbox_server.py execution_engine.py path_resolver.py /app/
WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "sandbox_server.py"]
