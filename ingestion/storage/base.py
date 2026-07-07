from abc import ABC, abstractmethod
from typing import Any


class BaseObjectStore(ABC):
    @abstractmethod
    def write(self, data: Any, path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def read(self, ref: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def exists(self, ref: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ref: str) -> None:
        raise NotImplementedError
