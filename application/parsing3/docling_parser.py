"""
Wraps Docling to turn PDF/DOCX/XLSX/image files into a unified structured
document, then uses Docling's own HybridChunker to split it into
embedding-ready chunks.

Why HybridChunker instead of a plain character-count splitter:
  - Works directly on the *structured* Docling document, so it respects
    headings, paragraphs, and table boundaries instead of cutting
    mid-sentence or mid-table.
  - Tokenization-aware: chunk sizes are measured in the actual tokens
    your embedding model will see, not raw character count.
  - Auto-merges undersized neighboring chunks sharing the same heading.
  - contextualize() prepends heading/caption context onto each chunk's
    text before embedding, improving retrieval on section-heavy docs.

FORMAT NOTES:
  - PDF/DOCX: text-layer based, OCR off by default (fast).
  - XLSX: uses Docling's own spreadsheet backend, no extra config needed.
  - Images (png/jpg/jpeg/tiff/bmp): OCR is always forced ON for images,
    regardless of the PDF OCR toggle, since an image has no underlying
    text layer at all — without OCR you'd get nothing back.

SPEED NOTES:
  - First call downloads Docling's ML models (~1-2GB) and caches them —
    that first run being slow is expected; later runs are much faster.
  - OCR/table-structure are OFF by default for PDFs (most uploaded PDFs
    already have selectable text). Flip via env vars only for scanned PDFs.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.chunking import HybridChunker
from docling_core.types.doc.document import DoclingDocument
from transformers import AutoTokenizer

from application.models.schemas import Chunk

# Must match the embedding model used downstream — keeps chunk sizing and
# embedding aligned to the same model's token limits.
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512

SUPPORTED_EXTENSIONS = {"pdf", "docx", "xlsx", "png", "jpg", "jpeg", "tiff", "bmp"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp"}

# Toggle via .env only for scanned/image-only PDFs — leave off for normal
# text-based PDFs/DOCX, which is what makes parsing fast.
ENABLE_OCR = os.environ.get("DOCLING_ENABLE_OCR", "false").lower() == "true"
ENABLE_TABLE_STRUCTURE = os.environ.get("DOCLING_ENABLE_TABLES", "false").lower() == "true"

_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
_chunker = HybridChunker(tokenizer=_tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)

_pdf_options = PdfPipelineOptions()
_pdf_options.do_ocr = ENABLE_OCR
_pdf_options.do_table_structure = ENABLE_TABLE_STRUCTURE
_pdf_options.generate_page_images = False
_pdf_options.generate_picture_images = False

# Images always need OCR — there's no text layer to fall back on.
_image_options = PdfPipelineOptions()
_image_options.do_ocr = True
_image_options.do_table_structure = ENABLE_TABLE_STRUCTURE
_image_options.generate_page_images = False
_image_options.generate_picture_images = False

# Built once at import time — reused across every parse call.
_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=_image_options),
        # XLSX and DOCX use Docling's default backends — no custom options needed.
    }
)


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    file_type: str
    docling_document: DoclingDocument
    full_text: str  # markdown export — handy for debugging/preview, not for embedding


def parse_document(file_path: str, doc_id: str) -> ParsedDocument:
    """Parse a PDF/DOCX/XLSX/image file into a unified, structure-preserving Docling document."""
    path = Path(file_path)
    file_type = path.suffix.lower().lstrip(".")

    if file_type not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '.{file_type}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    result = _converter.convert(file_path)
    docling_doc = result.document

    full_text = docling_doc.export_to_markdown()

    if file_type in IMAGE_EXTENSIONS and not full_text.strip():
        # Not fatal — some images genuinely have no text — but worth surfacing
        # since a silent empty chunk list downstream is confusing to debug.
        full_text = ""

    return ParsedDocument(
        doc_id=doc_id,
        filename=path.name,
        file_type=file_type,
        docling_document=docling_doc,
        full_text=full_text,
    )


def chunk_document(parsed: ParsedDocument) -> list[Chunk]:
    """
    Structure-aware, tokenization-aware chunking using Docling's HybridChunker.
    Returns a list of Chunk objects ready to be embedded and indexed.
    """
    chunk_iter = _chunker.chunk(dl_doc=parsed.docling_document)

    chunks: list[Chunk] = []
    for i, raw_chunk in enumerate(chunk_iter):
        text = _chunker.contextualize(chunk=raw_chunk)
        chunks.append(Chunk(
            chunk_id=f"{parsed.doc_id}_chunk_{i}",
            doc_id=parsed.doc_id,
            text=text,
            chunk_index=i,
        ))

    return chunks


def parse_and_chunk(file_path: str, doc_id: str) -> tuple[ParsedDocument, list[Chunk]]:
    """Convenience wrapper: parse then chunk in one call — what upload.py uses."""
    parsed = parse_document(file_path, doc_id)
    chunks = chunk_document(parsed)
    return parsed, chunks