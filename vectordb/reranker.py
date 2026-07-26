import logging
import time
from abc import ABC, abstractmethod

import requests

from config import get_settings

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models/{model}"

logger = logging.getLogger("vectordb.reranker")


class BaseReranker(ABC):
    @abstractmethod
    def rank(self, query: str, chunks: list, top_k: int = None) -> list:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """Reranks chunks using a cross-encoder model served through the
    Hugging Face (serverless) Inference API - no local model weights, no
    torch/CUDA in this image. Requires HF_API_TOKEN and network access to
    huggingface.co at request time.
    """

    def __init__(self, model_name: str = None, timeout: float = 30.0):
        settings = get_settings()
        self.model_name = model_name or settings.get("RERANKER_MODEL", DEFAULT_MODEL)
        self.api_token = settings.get("HF_API_TOKEN")
        if not self.api_token:
            raise RuntimeError(
                "HF_API_TOKEN is required to use the Hugging Face Inference API reranker"
            )
        self.url = HF_INFERENCE_URL.format(model=self.model_name)
        self.timeout = timeout

    def rank(self, query: str, chunks: list, top_k: int = None) -> list:
        if not chunks:
            return []

        start = time.perf_counter()
        payload = {
            "inputs": [{"text": query, "text_pair": chunk.text} for chunk in chunks],
            "options": {"wait_for_model": True},
        }
        response = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_token}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        results = response.json()

        scores = [self._extract_score(r) for r in results]
        ranked = [c for _, c in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
        logger.info(
            "reranker call took %.3fs (%d chunks in, %d returned)",
            time.perf_counter() - start, len(chunks), len(ranked[:top_k] if top_k else ranked),
        )
        return ranked[:top_k] if top_k else ranked

    @staticmethod
    def _extract_score(result) -> float:
        # text-classification returns [{"label": ..., "score": ...}] per
        # input pair; these cross-encoders emit a single label, so its score
        # is the relevance score.
        entry = result[0] if isinstance(result, list) else result
        return float(entry["score"])


class DeepInfraReranker(BaseReranker):
    """Reranks chunks using a Qwen3-Reranker model served through DeepInfra's inference
    API - no local model weights, no torch/CUDA in this image, same provider this codebase
    already uses for chat completions (see llm_provider/providers/deepinfra_client.py) and
    the shadow intent classifier (shared/intent_classifier.py). Requires DEEPINFRA_API_KEY
    and network access to deepinfra.com at request time.

    DeepInfra's rerank endpoint scores query/document PAIRS - `queries` and `documents` must
    be the same length (see https://deepinfra.com/Qwen/Qwen3-Reranker-4B/api) - there is no
    "one query, many documents" broadcast mode, so `rank()` repeats the query once per chunk
    rather than sending it once.
    """

    DEFAULT_MODEL = "Qwen/Qwen3-Reranker-4B"
    _URL = "https://api.deepinfra.com/v1/inference/{model}"

    def __init__(self, model_name: str = None, timeout: float = 30.0):
        settings = get_settings()
        self.model_name = model_name or settings.get("RERANKER_MODEL", self.DEFAULT_MODEL)
        self.api_token = settings.get("DEEPINFRA_API_KEY")
        if not self.api_token:
            raise RuntimeError(
                "DEEPINFRA_API_KEY is required to use the DeepInfra reranker"
            )
        self.url = self._URL.format(model=self.model_name)
        self.timeout = timeout

    def rank(self, query: str, chunks: list, top_k: int = None) -> list:
        if not chunks:
            return []

        start = time.perf_counter()
        payload = {
            "queries": [query] * len(chunks),
            "documents": [chunk.text for chunk in chunks],
        }
        response = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_token}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        scores = response.json()["scores"]

        ranked = [c for _, c in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
        logger.info(
            "deepinfra reranker call took %.3fs (%d chunks in, %d returned)",
            time.perf_counter() - start, len(chunks), len(ranked[:top_k] if top_k else ranked),
        )
        return ranked[:top_k] if top_k else ranked
