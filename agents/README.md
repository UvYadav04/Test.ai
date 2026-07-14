# Agents

Each agent gets its own folder: a `config.py` (system prompt + which LLM provider/model it uses) and an `agent.py` (the agent class itself, wiring the matching `tools/<category>` class into an autogen `AssistantAgent`).

- `tabular/` - `TabularAgent`, wraps `tools.tabular.TabularTools` + an autogen `AssistantAgent`. Runs its own tool-calling loop (plain-language finish, no JSON, no reflection), then hands a transcript of the tool calls/results to a second, tool-free `AssistantAgent` ("formatter") whose only job is to emit the final JSON. Splitting these two phases avoids models that try to emit JSON as a fake tool call when tools are still attached to the request.

```python
from agents.tabular import TabularAgent
from tools.tabular.models import FileRef

agent = TabularAgent([FileRef(file_id="f1", output_ref="data/parquet/ws1/f1.parquet", filename="f1.csv")])
findings = await agent.run("What is the average salary per department?")
```

- `document/` - `DocumentAgent`, wraps `tools.document.DocumentTools` (vector search over Chroma, optionally reranked) + the same tool-calling-then-formatter pattern as `tabular/`. Needs a `vector_store` (defaults to `ChromaVectorStore`, i.e. Chroma Cloud) and a `reranker` (defaults to `CrossEncoderReranker`) - pass your own instances in to reuse a connection across multiple agent runs. Tool loop: `get_file_overview` to orient, `search_documents`/`search_within_file` (+ `expand_query` if a search comes up thin) for narrow factual questions, `list_tables`/`search_tables`/`get_table` for anything table-related (these report a `table_ref` for the orchestrator to hand to the Tabular Agent, never row data), `broad_scan` for whole-document objectives a similarity search would only partially cover, and `verify_chunk_supports_claim`/`search_for_contradictions` before finalizing a confident claim.

```python
from agents.document import DocumentAgent
from tools.orchestrator.models import FileRef

agent = DocumentAgent([FileRef(file_id="handbook", output_ref="")])
findings = await agent.run("What is this document about?")
```

- `orchestrator/` - `OrchestratorAgent`, the top of the hierarchy: talks to the user, never runs SQL/RAG itself, delegates to fresh `TabularAgent`/`DocumentAgent` instances scoped to exactly the files it assigns them. Same tool-loop-then-formatter pattern, but returns `OrchestratorResult` (`final_answer`, `confidence`, `artifact_refs`, `open_questions`) instead of a `Findings` dataclass. A fresh `InvestigationState` (session_id, objective, selected_files, completed_tasks, findings, open_questions, status) is created per `run()` call - every `invoke_tabular_agent`/`invoke_document_agent`/`generate_hypotheses` call appends an event to it, and its `.summary()` feeds into the formatter alongside the transcript. `store_user_info`/`recall_user_info` are separate from Investigation State - they persist across sessions via `tools.orchestrator.memory.LongTermMemory` (a small JSON file), for facts that should outlive one investigation. Pass a `storage` instance (e.g. `LocalParquetStore`) to enable the deliverable tools below - without it, `generate_csv`/`generate_markdown_report`/`generate_dashboard` raise instead of running.

  Deliverable tools (`tools.reporting.ReportingTools`, wrapped by `OrchestratorTools`): `generate_csv` and `generate_dashboard` both read an existing Parquet artifact via its `output_ref` (from a Document Agent's `table_ref`, or from a Tabular Agent's `export_query` result) and turn it into a `.csv` or a single-file HTML dashboard (Chart.js from CDN, no server needed). `generate_markdown_report` takes the orchestrator's own synthesized `summary`/`findings`/`open_questions` text (not raw tool output) and writes a `.md` file. If the objective needs freshly computed data exported (not something already sitting in a file), the orchestrator tells the Tabular Agent to call `export_query` instead of `query_data`, which persists the full result as a new Parquet file and returns its `output_ref` - `query_data` alone only returns a row-capped preview, nothing is saved. Combining data across several files happens inside one `export_query` SQL call on the Tabular Agent (every assigned file is a queryable DuckDB view in the same connection), not by the orchestrator merging multiple agents' outputs itself. Every one of these three calls takes a `name` argument and creates `data/reports/<today's date>/<name>/` to hold the deliverable plus a copy of every source Parquet file it was built from, so each request's output is self-contained on disk.

  Model fallback: `OrchestratorAgent` and `DocumentAgent` both build their `LLMProvider` with `fallback_provider="groq"`, so `llm_provider.fallback_client.FallbackChatCompletionClient` transparently retries any failed `create()`/`create_stream()` call (e.g. a misconfigured or down primary provider) against a Groq client before giving up. For `DocumentAgent` this also covers `DocumentTools`' own bounded LLM calls (`expand_query`, contradiction/verification checks), since they share the same `LLMProvider` instance. `TabularAgent` does not get a fallback wired in currently.

```python
from agents.orchestrator import OrchestratorAgent
from tools.orchestrator.file_catalog import FileCatalog, entries_from_ingestion

catalog = FileCatalog()
for entry in entries_from_ingestion(ingestion_result, filename="employees.csv", file_type="csv"):
    catalog.add_entry(entry)

agent = OrchestratorAgent(catalog)
result = await agent.run("What is the average salary per department?", workspace_id="ws1")
```

Each agent is tested in isolation first (see `test_tabular_agent.py`, `test_document_agent.py`, and `test_orchestrator_agent.py` at the project root).

> Note: a Monitor Agent (log watchdog) and an evaluation harness (deepeval + Confident AI) were built and then removed for now - `agents/monitor/`, `tools/monitor/`, `evaluation/`, `run_monitor.py`, and `run_evaluation.py` are present but empty/inert. Safe to delete, or ask to have them rebuilt later.

## Logging

`logger.py` configures ONE shared logger (`logging.getLogger("agent")`) with a single `RotatingFileHandler` writing to `logs/agents.log` (plus console output), set up exactly once on first use. `get_agent_logger(name)` returns a child logger (`agent.<name>`) that propagates up to that shared handler, so every agent - tabular, document, orchestrator, hypothesis - writes into the same file instead of each owning a separate log. Level comes from `AGENT_LOG_LEVEL` in `.env` (defaults to `INFO`). `log_event(logger, event)` knows how to print autogen's streaming event types - what got sent to the agent, which tool it called with what arguments, and what came back. Every agent should log via `run_stream()` + `log_event`, not `agent.run()` + prints, so this stays consistent across agents:

```python
task_result = None
async for event in self.agent.run_stream(task=task):
    if hasattr(event, "messages"):
        task_result = event  # the final TaskResult
    else:
        log_event(self.logger, event)
```

## Add a new agent

Create a new folder here with `config.py` (system message + `get_model_config()`) and `agent.py` (a class that builds a `TabularTools`-style tool set, wraps it in an `AssistantAgent`, and parses the final reply into the matching `Findings` dataclass from `tools/orchestrator/models.py`).
