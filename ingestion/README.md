# Ingestion Pipeline

Turns uploaded files into a form the agents can query later.

- CSV / JSON -> Parquet, via `IngestionManager` + `LocalParquetStore`.
- PDF -> hybrid pipeline, text and tables handled separately (charts/images are skipped for now):
  - Parsing is done by **LlamaParse** (`llamaparse_client.py`), not local docling - docling's own CPU layout/table-structure pipeline measured 858.79s for one dense PDF, which was blowing past `run_ingestion`'s arq `job_timeout`. LlamaParse runs that work as a hosted API instead; requires `LLAMAPARSE_API_KEY` in `.env`. Its per-page markdown is then run through docling's lightweight Markdown backend (`InputFormat.MD` -> `SimplePipeline`, no ML models) purely to get a `DoclingDocument` per page for the chunker below - this is not the expensive path. `validate()` stays fully local/offline (`pypdf` structural check) so a bad upload doesn't burn a LlamaParse call.
  - Prose is chunked per page by `DoclingChunker` (docling's `HybridChunker`, keeps headings with their paragraphs) and stored in Chroma Cloud as plain text - Chroma embeds it automatically, no separate embedding step needed. Chunking is done per LlamaParse page (rather than one combined document) specifically to keep page numbers accurate, since markdown has no native page concept for docling to derive them from - LlamaParse's own page split is the source of truth.
  - Tables are pulled out via `extract_tables()` (docling's `document.tables`, called per page with `page_override` set to LlamaParse's true page number), each one written to its own Parquet file through the same `storage` backend (`file_id` = `"{file_id}_table_{n}"`, `n` running globally across all pages), so they're queryable by the Tabular Agent exactly like an ingested CSV. A short "pointer chunk" (caption + column names, metadata `{"type": "table", "page": n, "table_ref": table_file_id, "row_count": n, "columns": "a, b, c"}`) also goes into the vector store, so when the Document Agent's tools surface it, `table_ref` tells the Main Orchestrator which Parquet file to hand to the Tabular Agent for the actual computation. `PDFIngestor` now needs both a `storage` and a `vector_store` - `storage=None` skips table extraction (text-only) with a warning in `errors`.
  - Table captions use a 4-tier fallback chain (`_infer_caption()`): docling's own explicit caption, then a text snippet from the nearest preceding prose chunk (reuses `DoclingChunker`'s already-computed, page/heading-aware chunks rather than re-walking docling's raw document tree), then that chunk's section heading, then `"Table on page N"` if nothing else is available.

## Add a new file type

Create a folder under `ingestion/file_types/<type>/` with an ingestor class implementing `validate`, `extract_metadata`, `ingest`. Then add one line to `ingestion/registry.py`.

## Swap storage or vector store

Implement `BaseObjectStore` or `BaseVectorStore` and pass your class into `IngestionManager` instead of `LocalParquetStore` / `ChromaVectorStore`.

## Not done yet

- Scanned PDFs are OCR'd by LlamaParse itself now (this used to be a real gap when parsing ran through local docling with `do_ocr=False`). `is_scanned()` still flags them after the fact, via too-little extracted text, for informational metadata only - it no longer blocks/warns anything.
- Charts/images in PDFs are ignored - text and tables only.
- No migration for already-ingested PDFs - table chunks written before `row_count`/`columns` were added to their metadata won't have those fields until the file is deleted from the vector store and re-ingested.
