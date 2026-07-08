# Agents

Each agent gets its own folder: a `config.py` (system prompt + which LLM provider/model it uses) and an `agent.py` (the agent class itself, wiring the matching `tools/<category>` class into an autogen `AssistantAgent`).

- `tabular/` - `TabularAgent`, wraps `tools.tabular.TabularTools` + an autogen `AssistantAgent`. Runs its own tool-calling loop (plain-language finish, no JSON, no reflection), then hands a transcript of the tool calls/results to a second, tool-free `AssistantAgent` ("formatter") whose only job is to emit the final JSON. Splitting these two phases avoids models that try to emit JSON as a fake tool call when tools are still attached to the request.

```python
from agents.tabular import TabularAgent
from tools.tabular.models import FileRef

agent = TabularAgent([FileRef(file_id="f1", output_ref="data/parquet/ws1/f1.parquet", filename="f1.csv")])
findings = await agent.run("What is the average salary per department?")
```

- `document/` - `DocumentAgent`, wraps `tools.document.DocumentTools` (vector search over Chroma, optionally reranked) + the same tool-calling-then-formatter pattern as `tabular/`. Needs a `vector_store` (defaults to `ChromaVectorStore`, i.e. Chroma Cloud) and a `reranker` (defaults to `CrossEncoderReranker`) - pass your own instances in to reuse a connection across multiple agent runs.

```python
from agents.document import DocumentAgent
from tools.orchestrator.models import FileRef

agent = DocumentAgent([FileRef(file_id="handbook", output_ref="")])
findings = await agent.run("What is this document about?")
```

Each agent is tested in isolation first (see `test_tabular_agent.py` and `test_document_agent.py` at the project root) before being wired into the Main Orchestrator.

## Logging

`logger.py` has `get_agent_logger(name)` (a configured `logging.Logger`, level from `AGENT_LOG_LEVEL` in `.env`, defaults to `INFO`) and `log_event(logger, event)`, which knows how to print autogen's streaming event types - what got sent to the agent, which tool it called with what arguments, and what came back. Every agent should log via `run_stream()` + `log_event`, not `agent.run()` + prints, so this stays consistent across agents:

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
