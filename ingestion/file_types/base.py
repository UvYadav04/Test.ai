
from abc import ABC, abstractmethod
from typing import Optional

from ingestion.models import IngestionResult
from ingestion.storage.base import BaseObjectStore
from vectordb.base import BaseVectorStore


class BaseIngestor(ABC):

    def __init__(
        self,
        storage: Optional[BaseObjectStore] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        self.storage = storage
        self.vector_store = vector_store

    @abstractmethod
    def validate(self, file_path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract_metadata(self, file_path: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def ingest(self, file_path: str, workspace_id: str, file_id: str) -> IngestionResult:
        raise NotImplementedError
