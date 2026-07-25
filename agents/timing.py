"""Measures how long each tool call took, by pairing a ToolCallRequestEvent with its matching
ToolCallExecutionEvent in the same autogen event stream - see agents/tabular/agent.py,
agents/document/agent.py, agents/orchestrator/agent.py's run() loops, which all follow the same
`async for event in self.agent.run_stream(...)` shape and already call agents.logger.log_event
on every event; ToolCallTimer.record is meant to be called right alongside that.

Every provider client (llm_provider/providers/*.py) sets parallel_tool_calls=False, so within
one agent's own loop, tool call requests and their executions are always strictly sequential -
one open call at a time - which is what lets this track pending start times in a plain dict
keyed by tool name instead of needing a real call-id (autogen's ToolCallRequestEvent/
ToolCallExecutionEvent don't surface one - see the note in agents/events.py).
"""
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
                    # No matching request seen (e.g. timer created mid-stream) - nothing to
                    # diff against, skip rather than log a bogus duration.
                    continue
                status = "error" if getattr(res, "is_error", False) else "ok"
                self._logger.info("tool call %s took %.3fs (%s)", res.name, now - start, status)
