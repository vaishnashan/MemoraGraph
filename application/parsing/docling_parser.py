"""
Wraps Docling to turn PDF/DOCX files into a unified structured document,
then uses Docling's own HybridChunker to split it into embedding-ready
chunks.

Why HybridChunker instead of a plain character-count splitter (e.g. a
generic LangChain RecursiveCharacterTextSplitter):
  - It works directly on the *structured* Docling document, not flattened
    text, so it respects headings, paragraphs, and table boundaries instead
    of cutting mid-sentence or mid-table.
  - It's tokenization-aware: chunk sizes are measured in the actual tokens
    your embedding model will see, not raw character count, so nothing
    silently overflows the embedding model's context window.
  - It auto-merges undersized neighboring chunks that share the same
    heading, so you don't end up with a pile of tiny near-empty chunks.
  - contextualize() prepends the relevant heading/caption context onto each
    chunk's text before embedding, which measurably improves retrieval
    quality for section-heavy documents (lecture notes, reports, specs).

Day 2 task: get this working end-to-end on a few sample files before
touching anything else.
"""
from dataclasses import dataclass
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.types.doc.document import DoclingDocument
from transformers import AutoTokenizer

from application.models.schemas import Chunk
# Must match the embedding model used downstream in the embeddings module —
# keeping these in lockstep is what makes "tokenization-aware" meaningful.
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512  # safe context window for MiniLM-L6-v2


SUPPORTED_EXTENSIONS = {"pdf", "docx"}

# Loaded once at import time — reused across every parse/chunk call.
_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
_chunker = HybridChunker(tokenizer=_tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    file_type: str
    docling_document: DoclingDocument
    full_text: str  # markdown export — handy for debugging/preview, not for embedding


def parse_document(file_path: str, doc_id: str) -> ParsedDocument:
    """Parse a PDF or DOCX file into a unified, structure-preserving Docling document."""
    path = Path(file_path)
    file_type = path.suffix.lower().lstrip(".")

    if file_type not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '.{file_type}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    converter = DocumentConverter()
    result = converter.convert(file_path)
    docling_doc = result.document

    # Markdown export is just for humans/debugging — actual chunk text for
    # embedding comes from chunk_document() below via contextualize().
    full_text = docling_doc.export_to_markdown()

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
    Returns a list of Chunk objects (from app.models.schemas) ready to be
    embedded and indexed.
    """
    chunk_iter = _chunker.chunk(dl_doc=parsed.docling_document)

    chunks: list[Chunk] = []
    for i, raw_chunk in enumerate(chunk_iter):
        # contextualize() prepends the chunk's heading/caption trail so the
        # embedded text carries structural context — always use this over
        # raw_chunk.text when generating embeddings.
        text = _chunker.contextualize(chunk=raw_chunk)
        chunks.append(Chunk(
            chunk_id=f"{parsed.doc_id}_chunk_{i}",
            doc_id=parsed.doc_id,
            text=text,
            chunk_index=i,
        ))

    return chunks


def parse_and_chunk(file_path: str, doc_id: str) -> tuple[ParsedDocument, list[Chunk]]:
    """Convenience wrapper: parse then chunk in one call — what upload.py should use."""
    parsed = parse_document(file_path, doc_id)
    chunks = chunk_document(parsed)
    return parsed, chunks