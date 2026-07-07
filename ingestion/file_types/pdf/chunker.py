from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page: int


class BaseChunker(ABC):
    @abstractmethod
    def chunk_pages(self, pages_text: list) -> list:
        raise NotImplementedError


class SemanticChunker(BaseChunker):
    def __init__(self, max_chars: int = 1500):
        self.max_chars = max_chars

    def chunk_pages(self, pages_text: list) -> list:
        chunks = []
        index = 0

        for page_num, text in enumerate(pages_text, start=1):
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            heading = ""

            for para in paragraphs:
                is_heading = len(para) < 60 and "\n" not in para

                if is_heading:
                    heading = para
                    continue

                combined = f"{heading}\n{para}" if heading else para
                heading = ""

                if len(combined) <= self.max_chars:
                    chunks.append(Chunk(text=combined, chunk_index=index, page=page_num))
                    index += 1
                else:
                    start = 0
                    while start < len(combined):
                        window = combined[start:start + self.max_chars]
                        chunks.append(Chunk(text=window, chunk_index=index, page=page_num))
                        index += 1
                        start += self.max_chars

            if heading:
                chunks.append(Chunk(text=heading, chunk_index=index, page=page_num))
                index += 1

        return chunks
