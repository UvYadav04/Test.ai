import asyncio
import glob
import os
from datetime import datetime, timezone

from agents.orchestrator import OrchestratorAgent
from ingestion.manager import IngestionManager
from ingestion.storage.local_store import LocalParquetStore
from tools.orchestrator.file_catalog import FileCatalog, entries_from_ingestion
from tools.orchestrator.models import FileCatalogEntry
from vectordb.chroma_store import ChromaVectorStore
from vectordb.reranker import CrossEncoderReranker

QUESTIONS = [
    # "What data do we have in this workspace? List the files and their formats.",
    # "What is the average salary per department in the employees file?",
    "What is this PDF document about, and does it contain any tables?",
    # "Remember that I prefer dollar amounts `rounded to the nearest thousand in your answers.",
]


pdfs=["sec.gov_Archives_edgar_data_831001_000110465926042942_c-20260414xex99d1.htm.pdf"]
csvs=["people-100.csv"]

def find_pdf(root: str,pdf:str) -> str:
    pdfs = glob.glob(os.path.join(root,pdf))
    if not pdfs:
        raise FileNotFoundError("no .pdf file found in project root - add one and rerun")
    return pdfs[0]


async def main():
    root = os.path.dirname(os.path.abspath(__file__))
    storage = LocalParquetStore(root_dir=os.path.join(root, "data", "parquet"))
    vector_store = ChromaVectorStore()
    catalog = FileCatalog()

    csv_path = os.path.join(root,csvs[0])

    csv_manager = IngestionManager(storage=storage, vector_store=None)
    csv_result = csv_manager.ingest_file(csv_path, workspace_id="ws_test", file_id=csvs[0])
    print("csv ingestion status:", csv_result.status, csv_result.errors)
    assert csv_result.status == "success"
    for entry in entries_from_ingestion(csv_result, filename="employees.csv", file_type="csv"):
        catalog.add_entry(entry)

    pdf_path = find_pdf(root,pdfs[0])
    pdf_file_id = os.path.splitext(os.path.basename(pdf_path))[0].replace(" ", "_")
    print("using pdf:", pdf_path)

    # vector_store.delete(ids=[])

    existing = vector_store.get_by_filter({"file_id": pdf_file_id})
    if existing:
        print(f"found {len(existing)} existing chunks for '{pdf_file_id}', skipping pdf ingestion")
        catalog.add_entry(FileCatalogEntry(
            file_id=pdf_file_id,
            filename=os.path.basename(pdf_path),
            file_type="pdf",
            uploaded_at=datetime.now(timezone.utc),
            size_bytes=os.path.getsize(pdf_path),
        ))
    else:
        print("ingesting pdf (text chunks + tables)...")
        pdf_manager = IngestionManager(storage=storage, vector_store=vector_store)
        pdf_result = pdf_manager.ingest_file(pdf_path, workspace_id="ws_test", file_id=pdf_file_id)
        print("pdf ingestion status:", pdf_result.status, pdf_result.errors, "chunks:", pdf_result.chunk_count)
        assert pdf_result.status in ("success", "partial")
        for entry in entries_from_ingestion(pdf_result, filename=os.path.basename(pdf_path), file_type="pdf"):
            catalog.add_entry(entry)

    print("\ncatalog entries:", [e.file_id for e in catalog.all()])

    reranker = CrossEncoderReranker()
    agent = OrchestratorAgent(catalog, vector_store=vector_store, reranker=reranker, storage=storage)

    while(True):
        input_ref = input("objective: ")
        if input_ref == "exit":
            break
        result = await agent.run(objective=input_ref, workspace_id="ws_test")

        print("\nfinal_answer:", result.final_answer)
        print("confidence:", result.confidence)
        print("artifact_refs:", result.artifact_refs)
        print("open_questions:", result.open_questions)


if __name__ == "__main__":
    asyncio.run(main())
