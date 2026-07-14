import logging
import os
from logging.handlers import RotatingFileHandler

from config import get_settings

_ROOT_NAME = "agent"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "agents.log")

_configured = False


def _configure_root() -> logging.Logger:
    global _configured
    root = logging.getLogger(_ROOT_NAME)
    if _configured:
        return root

    os.makedirs(LOG_DIR, exist_ok=True)
    settings = get_settings()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    root.setLevel(settings.get("AGENT_LOG_LEVEL", "INFO") or "INFO")
    root.propagate = False
    _configured = True
    return root


def get_agent_logger(name: str) -> logging.Logger:
    """Every agent's logger is a child of one shared "agent" logger, so all agent activity
    (tabular, document, orchestrator, hypothesis, monitor, ...) lands in the single file at
    logs/agents.log (plus console), instead of each agent owning its own separate log file."""
    _configure_root()
    logger = logging.getLogger(f"{_ROOT_NAME}.{name}")
    logger.propagate = True
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
