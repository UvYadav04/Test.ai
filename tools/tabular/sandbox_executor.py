"""Host-side driver for running model-generated pandas/DuckDB code inside an isolated Docker
container (see sandbox/Dockerfile, sandbox/runner.py). The container has no network access, a
capped memory/CPU allowance, and a hard wall-clock timeout - it can only read the parquet files
under the app's own storage root and write new ones there via the runner's save() helper. Only
whatever the runner explicitly returns (stdout, capped; describe()/preview()/save() outputs)
ever leaves the container - the full DataFrames never do.
"""
import json
import os
import shutil
import tempfile

import docker
from docker.errors import ImageNotFound

IMAGE_NAME = "dataanalyzer-sandbox:latest"
_SANDBOX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sandbox")


class SandboxExecutionError(RuntimeError):
    pass


class PythonSandbox:
    def __init__(
        self,
        root_dir: str,
        image: str = IMAGE_NAME,
        timeout_seconds: int = 30,
        mem_limit: str = "512m",
        nano_cpus: int = 1_000_000_000,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                self._client = docker.from_env()
            except Exception as exc:
                raise SandboxExecutionError(
                    "could not connect to Docker - is Docker Desktop/daemon running?"
                ) from exc
        return self._client

    def ensure_image(self) -> None:
        try:
            self.client.images.get(self.image)
        except ImageNotFound:
            self.client.images.build(path=_SANDBOX_DIR, tag=self.image, rm=True)

    def run(self, code: str, tables: dict, workspace_id: str) -> dict:
        """tables: {table_name: host_output_ref}. Every output_ref must live under root_dir -
        that's the only thing bind-mounted into the container."""
        self.ensure_image()

        container_tables = {}
        for table_name, output_ref in tables.items():
            abs_ref = os.path.abspath(output_ref)
            if os.path.commonpath([abs_ref, self.root_dir]) != self.root_dir:
                raise SandboxExecutionError(
                    f"file for table '{table_name}' is not under the sandbox's data root "
                    f"({self.root_dir}) - refusing to mount it"
                )
            rel = os.path.relpath(abs_ref, self.root_dir).replace(os.sep, "/")
            container_tables[table_name] = f"/data/{rel}"

        job_dir = tempfile.mkdtemp(prefix="sandbox_job_")
        container = None
        try:
            manifest = {"tables": container_tables, "workspace_id": workspace_id, "code": code}
            with open(os.path.join(job_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            container = self.client.containers.run(
                self.image,
                detach=True,
                network_disabled=True,
                mem_limit=self.mem_limit,
                nano_cpus=self.nano_cpus,
                volumes={
                    self.root_dir: {"bind": "/data", "mode": "rw"},
                    job_dir: {"bind": "/job", "mode": "rw"},
                },
            )

            timed_out = False
            try:
                container.wait(timeout=self.timeout_seconds)
            except Exception:
                timed_out = True
                try:
                    container.kill()
                except Exception:
                    pass

            try:
                logs = container.logs().decode("utf-8", errors="replace")
            except Exception:
                logs = ""

            result_path = os.path.join(job_dir, "result.json")
            if not os.path.exists(result_path):
                return {
                    "stdout": logs[-2000:],
                    "saved": [],
                    "error": (
                        f"sandbox timed out after {self.timeout_seconds}s"
                        if timed_out else "sandbox exited without producing a result"
                    ),
                }

            with open(result_path, encoding="utf-8") as f:
                raw_result = json.load(f)

            for entry in raw_result.get("saved", []):
                container_path = entry["output_ref"]
                rel = container_path[len("/data/"):] if container_path.startswith("/data/") else container_path
                entry["output_ref"] = os.path.join(self.root_dir, rel.replace("/", os.sep))

            return raw_result
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
            shutil.rmtree(job_dir, ignore_errors=True)
