import asyncio
import os
import tempfile

from agents.tabular import TabularAgent
from ingestion.manager import IngestionManager
from ingestion.storage.local_store import LocalParquetStore
from tools.tabular.models import FileRef

SAMPLE_CSV = """name,department,salary,years_experience,hire_date
Alice,Engineering,95000,4,2021-03-01
Bob,Sales,62000,2,2022-07-15
Carol,Engineering,105000,7,2018-11-20
Dave,Sales,58000,1,2023-01-10
Eve,Marketing,72000,3,2020-06-05
Frank,Engineering,88000,2,2022-09-01
Grace,Marketing,79000,5,2019-04-18
Heidi,Sales,64000,3,2021-12-02
Ivan,Engineering,112000,9,2016-02-14
Judy,Marketing,67000,1,2023-05-30
"""

QUESTIONS = [
    "What is the average salary per department?",
    "Who are the top 3 highest paid employees?",
    "Is there a relationship between years_experience and salary?",
    "Which department has the most employees, and how many?",
    "Are there any employees hired before 2019? List them.",
]


async def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        print("tmp_dir: ",tmp_dir)
        csv_path = os.path.join(tmp_dir, "employees.csv")
        print("csv_path: ",csv_path)
        with open(csv_path, "w") as f:
            f.write(SAMPLE_CSV)

        print("csv_path: ",csv_path)

        storage = LocalParquetStore(root_dir=os.path.join(tmp_dir, "parquet"))
        manager = IngestionManager(storage=storage, vector_store=None)

        print("ingesting employees.csv...")
        result = manager.ingest_file(csv_path, workspace_id="ws_test", file_id="employees")
        print("ingestion status:", result.status, result.errors)
        assert result.status == "success"

        assigned_files = [FileRef(file_id="employees", output_ref=result.output_ref, filename="employees.csv")]

        agent = TabularAgent(assigned_files)

        for question in QUESTIONS:
            print("\n" + "=" * 80)
            print("objective:", question)
            findings = await agent.run(objective=question)

            print("\nsummary:", findings.summary)
            print("artifact_refs:", findings.artifact_refs)


if __name__ == "__main__":
    asyncio.run(main())
