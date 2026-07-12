# Deliberately no eager re-exports here: orchestrator_tools.py imports from agents.tabular
# and agents.document (to spawn them), which is the reverse of every other tools/<category>
# package. Eagerly importing OrchestratorTools/FileCatalog at package-init time risks a
# circular import depending on what gets imported first - use the direct submodule paths
# instead, e.g. `from tools.orchestrator.orchestrator_tools import OrchestratorTools`,
# `from tools.orchestrator.file_catalog import FileCatalog`.
