import asyncio
import glob
import os

from agents.document import DocumentAgent
from ingestion.manager import IngestionManager
from ingestion.storage.local_store import LocalParquetStore
from tools.orchestrator.models import FileRef
from vectordb.chroma_store import ChromaVectorStore
from vectordb.reranker import CrossEncoderReranker

QUESTIONS = [
    "What is this document about? Give a high-level summary.",
    "What are the main sections or topics covered in this document?",
    "What are the most important facts, numbers, or conclusions mentioned in this document?",
    "Does this document contain any tables? If so, what do they show?",
]


def find_pdf() -> str:
    print("finding pdf")
    root = os.path.dirname(os.path.abspath(__file__))
    pdfs = glob.glob(os.path.join(root, "*.pdf"))
    print("found pdf")
    if not pdfs:
        raise FileNotFoundError("no .pdf file found in project root - add one and rerun")
    return pdfs[0]


async def main():
    pdf_path = find_pdf()
    file_id = os.path.splitext(os.path.basename(pdf_path))[0].replace(" ", "_")
    print("using pdf:", pdf_path)
    print("file_id:", file_id)

    vector_store = ChromaVectorStore()

    existing = vector_store.get_by_filter({"file_id": file_id})
    if existing:
        print(f"found {len(existing)} existing chunks for '{file_id}', skipping ingestion")
    else:
        print("ingesting pdf (text chunks + tables)...")
        root = os.path.dirname(os.path.abspath(__file__))
        storage = LocalParquetStore(root_dir=os.path.join(root, "data", "parquet"))
        manager = IngestionManager(storage=storage, vector_store=vector_store)
        result = manager.ingest_file(pdf_path, workspace_id="ws_test", file_id=file_id)
        print("ingestion status:", result.status, result.errors, "chunks:", result.chunk_count)
        print("extracted tables:", result.extracted_tables)
        assert result.status in ("success", "partial")

    assigned_files = [FileRef(file_id=file_id, output_ref="")]
    reranker = CrossEncoderReranker()
    agent = DocumentAgent(assigned_files, vector_store=vector_store, reranker=reranker)

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print("objective:", question)
        findings = await agent.run(objective=question)

        print("\nsummary:", findings.summary)
        print("findings:", findings.findings)
        print("limitations:", findings.limitations)
        print("confidence:", findings.confidence)
        print("source_refs:", findings.source_refs)
        print("artifact_refs (table_refs, if any):", findings.artifact_refs)


if __name__ == "__main__":
    asyncio.run(main())
