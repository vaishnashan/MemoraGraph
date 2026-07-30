"""
Wraps Docling to turn PDF/DOCX files into a unified structured document,
then splits that into overlapping text chunks ready for embedding.

Day 2 task: get this working end-to-end on a few sample files before
touching anything else.
"""
from dataclasses import dataclass
from docling.document_converter import DocumentConverter


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    file_type: str
    full_text: str


def parse_document(file_path: str, doc_id: str) -> ParsedDocument:
    """Parse a PDF or DOCX file into unified structured text."""
    converter = DocumentConverter()
    result = converter.convert(file_path)

    # Docling exposes a unified document object; export to markdown
    # to preserve headings/tables/structure in a text-friendly form.
    full_text = result.document.export_to_markdown()

    filename = file_path.split("/")[-1]
    file_type = filename.split(".")[-1].lower()

    return ParsedDocument(
        doc_id=doc_id,
        filename=filename,
        file_type=file_type,
        full_text=full_text,
    )


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Simple sliding-window chunking by character count.
    chunk_size/overlap are tunable — start here, tune once you see
    real retrieval quality on your golden eval set (Week 3).
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap  # step forward, keeping overlap

    return chunks
