from config import get_settings


def get_model_config() -> dict:
    settings = get_settings()
    return {
        "provider": settings.get("HYPOTHESIS_PROVIDER", "") or None,
        "model": settings.get("HYPOTHESIS_MODEL", "") or None,
    }
