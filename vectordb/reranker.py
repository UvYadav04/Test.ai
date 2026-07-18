from abc import ABC, abstractmethod

import requests

from config import get_settings

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models/{model}"


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
        return ranked[:top_k] if top_k else ranked

    @staticmethod
    def _extract_score(result) -> float:
        # text-classification returns [{"label": ..., "score": ...}] per
        # input pair; these cross-encoders emit a single label, so its score
        # is the relevance score.
        entry = result[0] if isinstance(result, list) else result
        return float(entry["score"])
