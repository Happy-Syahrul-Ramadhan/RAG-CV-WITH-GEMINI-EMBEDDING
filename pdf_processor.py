"""
PDF Processor Module.

Handles PDF text extraction using PyMuPDF and text chunking
for embedding. For short PDFs (≤6 pages), the raw bytes can
be sent directly to Gemini-embedding-2 for native multimodal
processing. For longer PDFs, text is extracted and chunked.
"""

import os
from typing import List, Tuple
import fitz  # PyMuPDF


class PDFProcessor:
    """Process PDF documents for embedding."""

    @staticmethod
    def read_pdf_bytes(pdf_path: str) -> bytes:
        """Read a PDF file as raw bytes."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        with open(pdf_path, "rb") as f:
            return f.read()

    @staticmethod
    def get_page_count(pdf_path: str) -> int:
        """Get the number of pages in a PDF."""
        doc = fitz.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count

    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """
        Extract all text from a PDF document.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Full text content of the PDF.
        """
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                full_text += f"\n\n--- Page {page_num} ---\n\n{text}"
        doc.close()
        return full_text.strip()

    @staticmethod
    def extract_text_by_page(pdf_path: str) -> List[Tuple[int, str]]:
        """
        Extract text from each page of a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of (page_number, text) tuples.
        """
        doc = fitz.open(pdf_path)
        pages = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text().strip()
            if text:
                pages.append((page_num, text))
        doc.close()
        return pages

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Full text to chunk.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between chunks.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position
            end = start + chunk_size

            # If this is the last chunk, take the rest
            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            # Try to break at a sentence or paragraph boundary
            # Look backwards for newline, period, or space
            for delimiter in ["\n\n", "\n", ". ", " "]:
                # Find last occurrence of delimiter within chunk
                last_boundary = text.rfind(delimiter, start, end)
                if last_boundary > start + chunk_size // 2:
                    end = last_boundary + len(delimiter)
                    break

            chunks.append(text[start:end].strip())

            # Move start position with overlap
            start = end - chunk_overlap

        return chunks

    @staticmethod
    def suggest_approach(pdf_path: str) -> str:
        """
        Suggest whether to use native PDF embedding or text extraction.

        Gemini-embedding-2 supports native PDF input but is limited
        to max 6 pages per request.
        """
        page_count = PDFProcessor.get_page_count(pdf_path)
        if page_count <= 6:
            return "native"
        else:
            return "text_extract"
