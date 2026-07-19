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


class FixedSizeChunker(BaseChunker):
    """Plain sliding-window chunker: fixed character length with overlap, no layout/heading
    awareness. Used by TXTIngestor (plain text has no page/heading structure for
    DoclingChunker to key off) and swappable into PDFIngestor for the same reason."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document) -> list:
        """`document` is either a docling document (export_to_text() is called first) or
        already a plain string."""
        text = document.export_to_text() if hasattr(document, "export_to_text") else str(document)
        text = text.strip()
        if not text:
            return []

        step = self.chunk_size - self.overlap
        chunks = []
        index = 0
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(Chunk(text=text[start:end], chunk_index=index, page=0))
            index += 1
            if end == len(text):
                break
            start += step
        return chunks
