# Ingestion Pipeline

Turns uploaded files into a form the agents can query later.

- CSV / JSON -> Parquet, via `IngestionManager` + `LocalParquetStore`.
- PDF -> chunked (by paragraph/heading, see `SemanticChunker`) and stored in Chroma.

## Add a new file type

Create a folder under `ingestion/file_types/<type>/` with an ingestor class implementing `validate`, `extract_metadata`, `ingest`. Then add one line to `ingestion/registry.py`.

## Swap storage or vector store

Implement `BaseObjectStore` or `BaseVectorStore` and pass your class into `IngestionManager` instead of `LocalParquetStore` / `ChromaVectorStore`.

## Not done yet

- No OCR for scanned PDFs.
- No embedding model wired in - `ChunkRecord.embedding` stays `None` until that's added.
