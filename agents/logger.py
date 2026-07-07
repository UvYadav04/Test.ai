import logging

from config import get_settings


def get_agent_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"agent.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False

        settings = get_settings()
        logger.setLevel(settings.get("AGENT_LOG_LEVEL", "INFO") or "INFO")

    return logger


def log_event(logger: logging.Logger, event) -> None:
    event_type = type(event).__name__

    if event_type == "TextMessage":
        logger.info("[%s] %s", event.source, event.content)
    elif event_type == "ToolCallRequestEvent":
        for call in event.content:
            logger.info("[tool call] %s(%s)", call.name, call.arguments)
    elif event_type == "ToolCallExecutionEvent":
        for res in event.content:
            status = "error" if res.is_error else "ok"
            logger.info("[tool result:%s] %s -> %s", status, res.name, res.content)
    else:
        logger.info("[%s] %s", event_type, getattr(event, "content", ""))
