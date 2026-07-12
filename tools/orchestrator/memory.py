import json
import os
from datetime import datetime, timezone


class LongTermMemory:
    def __init__(self, path: str = "data/memory/long_term.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self._write([])

    def remember(self, info: str) -> None:
        entries = self._read()
        entries.append({"info": info, "stored_at": datetime.now(timezone.utc).isoformat()})
        self._write(entries)

    def recall_all(self) -> list:
        return [e["info"] for e in self._read()]

    def _read(self) -> list:
        with open(self.path) as f:
            return json.load(f)

    def _write(self, entries: list) -> None:
        with open(self.path, "w") as f:
            json.dump(entries, f, indent=2)
