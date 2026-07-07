from pypdf import PdfReader


def extract_text_per_page(file_path: str) -> list:
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages


def is_scanned(pages: list) -> bool:
    if not pages:
        return False
    avg_chars = sum(len(p.strip()) for p in pages) / len(pages)
    return avg_chars < 20
