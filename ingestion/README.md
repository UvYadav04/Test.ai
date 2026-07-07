# Ingestion Pipeline

Turns uploaded files into a form the agents can query later.

- CSV / JSON -> Parquet, via `IngestionManager` + `LocalParquetStore`.
- PDF -> parsed and chunked by `DoclingChunker` (docling's `HybridChunker`, keeps headings with their paragraphs) and stored in Chroma Cloud as plain text - Chroma embeds it automatically with its bundled default model, no separate embedding step needed.

## Add a new file type

Create a folder under `ingestion/file_types/<type>/` with an ingestor class implementing `validate`, `extract_metadata`, `ingest`. Then add one line to `ingestion/registry.py`.

## Swap storage or vector store

Implement `BaseObjectStore` or `BaseVectorStore` and pass your class into `IngestionManager` instead of `LocalParquetStore` / `ChromaVectorStore`.

## Not done yet

- No OCR for scanned PDFs.
