from abc import ABC, abstractmethod

from sentence_transformers import CrossEncoder

from config import get_settings

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class BaseReranker(ABC):
    @abstractmethod
    def rank(self, query: str, chunks: list, top_k: int = None) -> list:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = None):
        settings = get_settings()
        self.model = CrossEncoder(model_name or settings.get("RERANKER_MODEL", DEFAULT_MODEL))

    def rank(self, query: str, chunks: list, top_k: int = None) -> list:
        if not chunks:
            return []

        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)

        ranked = [c for _, c in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
        return ranked[:top_k] if top_k else ranked
