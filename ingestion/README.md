# Ingestion Pipeline

Turns uploaded files into a form the agents can query later.

- CSV / JSON -> Parquet, via `IngestionManager` + `LocalParquetStore`.
- PDF -> hybrid pipeline, text and tables handled separately (charts/images are skipped for now):
  - Prose is chunked by `DoclingChunker` (docling's `HybridChunker`, keeps headings with their paragraphs) and stored in Chroma Cloud as plain text - Chroma embeds it automatically, no separate embedding step needed.
  - Tables are pulled out via `extract_tables()` (docling's `document.tables`), each one written to its own Parquet file through the same `storage` backend (`file_id` = `"{file_id}_table_{n}"`), so they're queryable by the Tabular Agent exactly like an ingested CSV. A short "pointer chunk" (caption + column names, metadata `{"type": "table", "page": n, "table_ref": table_file_id}`) also goes into the vector store, so when the Document Agent's RAG search surfaces it, `table_ref` tells the Main Orchestrator which Parquet file to hand to the Tabular Agent for the actual computation. `PDFIngestor` now needs both a `storage` and a `vector_store` - `storage=None` skips table extraction (text-only) with a warning in `errors`.

## Add a new file type

Create a folder under `ingestion/file_types/<type>/` with an ingestor class implementing `validate`, `extract_metadata`, `ingest`. Then add one line to `ingestion/registry.py`.

## Swap storage or vector store

Implement `BaseObjectStore` or `BaseVectorStore` and pass your class into `IngestionManager` instead of `LocalParquetStore` / `ChromaVectorStore`.

## Not done yet

- No OCR for scanned PDFs - `convert_document()` runs docling with `do_ocr=False` on purpose (born-digital PDFs only for now, and this skips a slow OCR-engine download/init on every conversion). `is_scanned()` still flags scanned PDFs after the fact, via too-little extracted text.
- Charts/images in PDFs are ignored - text and tables only.
