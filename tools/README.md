# Tools

Tools each agent can call, grouped by agent type in their own folder.

- `tabular/` - Tabular Agent tools (`TabularTools`), backed by DuckDB over the Parquet files ingestion produces. Scoped to whatever `FileRef` list the agent was assigned.
- `document/` - Document Agent tools (`DocumentTools`), backed by the vector store. Needs a `vector_store`, an `embed_fn` (text -> vector, plugs in once an embedding model is wired up) and optionally a `CrossEncoderReranker`.
- `orchestrator/` - Main Orchestrator tools (`OrchestratorTools`), backed by an in-memory `FileCatalog`. `invoke_tabular_agent`/`invoke_document_agent` are thin dispatchers - pass in the real agent loops once they exist.
- `hypothesis/` - `HypothesisTools.generate_hypotheses`, a single bounded LLM call (no tools, no loop) used by the orchestrator before delegating on complex "why" queries.

`llm_call.py` is a small shared helper (`ask_llm(client, prompt)`) used by anything that needs one bounded LLM call - `document/` and `hypothesis/` both use it.

## Add a new tool category

Create a new folder here with its own `models.py` and a `<category>_tools.py` class, following the same shape as the folders above: constructed with the files/context it's allowed to touch, one method per tool.
