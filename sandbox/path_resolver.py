"""Single source of truth for how a (workspace_id, artifact_id) pair maps to a physical
parquet path, on BOTH sides of the sandbox boundary.

Deliberately pure stdlib (os, re, uuid) - this exact file is COPYd verbatim into the sandbox
Docker image (see Dockerfile) so runner.py can import it without adding anything to that
image's dependency/attack surface, and imported normally by host-side code (sandbox_executor,
tabular_tools, orchestrator_tools, reporting_tools, worker_service tasks, the ingestors) as
`sandbox.path_resolver`. One file, one convention, both sides derive it independently instead
of a host path being computed once and threaded/rewritten across the container boundary (see
storage-refactor-spec.md and outputref-vulnerability-audit.md for why that used to happen).

Nothing here ever accepts a caller-supplied path. Every function takes a workspace_id and/or
artifact_id (both validated against `_SAFE_SEGMENT_RE`) and builds a path from them - that
validation is the only guard against path traversal anywhere a workspace_id/artifact_id
crosses a trust boundary (an LLM-supplied file_id, a Mongo doc, a tool argument).
"""
import os
import re
import uuid

# Matches docker-compose.yml's `parquet_data` named volume, mounted at this exact path in
# BOTH worker_service (this process) and every sandbox container it spins up via
# docker-outside-of-docker - see PARQUET_VOLUME_MOUNT there and the note in sandbox_executor.py.
SANDBOX_ROOT = "/data/parquet"

_SAFE_SEGMENT_RE = re.compile(r"^[0-9a-zA-Z_-]+$")


class InvalidArtifactIdError(ValueError):
    


def validate_segment(value: str, label: str = "id") -> str:
    """Reject anything that isn't a plain [0-9a-zA-Z_-]+ token - no '/', '\\', '..', or empty
    string. Every path-building function below calls this on each segment before touching
    os.path.join/f-strings, so a malformed workspace_id or artifact_id fails loudly here
    instead of silently resolving outside the intended directory."""
    if not isinstance(value, str) or not value or not _SAFE_SEGMENT_RE.match(value):
        raise InvalidArtifactIdError(
            f"invalid {label} {value!r} - must be a non-empty string of letters, digits, "
            "underscore, or hyphen only (no '/', '\\', or '..')"
        )
    return value


def get_parquet_path(root_dir: str, workspace_id: str, artifact_id: str) -> str:
    """Absolute path to a parquet artifact as seen from whichever container's OWN mount of
    the shared parquet volume `root_dir` points at (worker_service: PARQUET_ROOT env var,
    == SANDBOX_ROOT in practice - see engine_bootstrap.py). Callers never pass a path in -
    only workspace_id/artifact_id, both validated - so this can't be pointed outside root_dir."""
    validate_segment(workspace_id, "workspace_id")
    validate_segment(artifact_id, "artifact_id")
    return os.path.join(os.path.abspath(root_dir), workspace_id, f"{artifact_id}.parquet")


def get_sandbox_path(workspace_id: str, artifact_id: str, sandbox_root: str = SANDBOX_ROOT) -> str:
    """Path to the SAME artifact as the sandbox container's own mount of the identical named
    Docker volume sees it. Both the host (sandbox_executor.py, building the manifest) and the
    sandbox itself (runner.py, loading tables / writing save() output) call this independently
    - no host-path arithmetic (abspath/relpath/commonpath) crosses the container boundary."""
    validate_segment(workspace_id, "workspace_id")
    validate_segment(artifact_id, "artifact_id")
    return f"{sandbox_root}/{workspace_id}/{artifact_id}.parquet"


def get_table_path(file_id: str) -> str:
    """DuckDB-safe view/table identifier derived from a file_id - formalizes the
    (file_id -> SQL identifier) mapping duckdb_utils.safe_view_name used to own alone, so
    it's defined in exactly one place shared by ArtifactStore-style callers too."""
    name = re.sub(r"[^0-9a-zA-Z_]", "_", file_id or "")
    if not name or name[0].isdigit():
        name = f"t_{name}"
    return name


def new_artifact_id(name: str = "result") -> str:
    """A short, filesystem-and-SQL-safe id for a freshly created artifact (run_python's
    save(), a persisted query_data/aggregate result). Replaces the ad hoc
    `f"{safe_name}_{uuid4().hex[:8]}"` that used to be typed out separately at every call site
    (sandbox/runner.py, tabular_tools.py, reporting - each with its own slightly different
    sanitization regex/length cap)."""
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))[:40].strip("_") or "result"
    return f"{safe}_{uuid.uuid4().hex[:8]}"
