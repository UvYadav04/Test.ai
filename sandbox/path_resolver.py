import os
import re
import uuid

SANDBOX_ROOT = "/data/parquet"

_SAFE_SEGMENT_RE = re.compile(r"^[0-9a-zA-Z_-]+$")


class InvalidArtifactIdError(ValueError):
    pass


def validate_segment(value: str, label: str = "id") -> str:
    if not isinstance(value, str) or not value or not _SAFE_SEGMENT_RE.match(value):
        raise InvalidArtifactIdError(
            f"invalid {label} {value!r} - must be a non-empty string of letters, digits, "
            "underscore, or hyphen only (no '/', '\\', or '..')"
        )
    return value


def get_parquet_path(root_dir: str, workspace_id: str, artifact_id: str) -> str:
    validate_segment(workspace_id, "workspace_id")
    validate_segment(artifact_id, "artifact_id")
    return os.path.join(os.path.abspath(root_dir), workspace_id, f"{artifact_id}.parquet")


def get_sandbox_path(workspace_id: str, artifact_id: str, sandbox_root: str = SANDBOX_ROOT) -> str:
    validate_segment(workspace_id, "workspace_id")
    validate_segment(artifact_id, "artifact_id")
    return f"{sandbox_root}/{workspace_id}/{artifact_id}.parquet"


def get_table_path(file_id: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]", "_", file_id or "")
    if not name or name[0].isdigit():
        name = f"t_{name}"
    return name


def new_artifact_id(name: str = "result") -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))[:40].strip("_") or "result"
    return f"{safe}_{uuid.uuid4().hex[:8]}"
