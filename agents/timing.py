import time


class ToolCallTimer:
    def __init__(self, logger):
        self._logger = logger
        self._pending: dict[str, float] = {}

    def record(self, event) -> None:
        event_type = type(event).__name__
        if event_type == "ToolCallRequestEvent":
            now = time.perf_counter()
            for call in event.content:
                self._pending[call.name] = now
        elif event_type == "ToolCallExecutionEvent":
            now = time.perf_counter()
            for res in event.content:
                start = self._pending.pop(res.name, None)
                if start is None:
                    continue
                status = "error" if getattr(res, "is_error", False) else "ok"
                self._logger.info("tool call %s took %.3fs (%s)", res.name, now - start, status)
