import os

from dotenv import dotenv_values, load_dotenv

load_dotenv()


class Settings:
    def __init__(self, values: dict):
        self._values = values

    def __getattr__(self, key):
        return self._values.get(key, os.getenv(key, ""))

    def get(self, key, default=""):
        return self._values.get(key, os.getenv(key, default))


_settings = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings(dotenv_values())
    return _settings
