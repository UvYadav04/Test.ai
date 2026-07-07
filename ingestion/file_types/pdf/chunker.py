from abc import ABC, abstractmethod
from dataclasses import dataclass

from docling.chunking import HybridChunker


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page: int
    section: str = ""


class BaseChunker(ABC):
    @abstractmethod
    def chunk_document(self, document) -> list:
        raise NotImplementedError


class DoclingChunker(BaseChunker):
    def __init__(self):
        self.chunker = HybridChunker()

    def chunk_document(self, document) -> list:
        chunks = []
        for index, chunk in enumerate(self.chunker.chunk(document)):
            headings = getattr(chunk.meta, "headings", None) or []
            chunks.append(Chunk(
                text=self._text_of(chunk, headings),
                chunk_index=index,
                page=self._page_of(chunk),
                section=headings[-1] if headings else "",
            ))
        return chunks

    def _text_of(self, chunk, headings: list) -> str:
        if headings:
            return "\n".join(headings) + "\n" + chunk.text
        return chunk.text

    def _page_of(self, chunk) -> int:
        try:
            return chunk.meta.doc_items[0].prov[0].page_no
        except Exception:
            return 0
