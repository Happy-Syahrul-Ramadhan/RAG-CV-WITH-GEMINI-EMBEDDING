"""
Embedding Service Module.

Handles communication with the Google Gemini Embedding API
using the google-genai SDK.
"""

from typing import List, Optional

from google import genai
from google.genai import types

from config import Config


class GeminiGenerationService:
    """
    Service for generating text using Gemini models (for RAG / Q&A).
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or Config.GEMINI_API_KEY
        if not key:
            raise ValueError(
                "Gemini API key is required. "
                "Set GEMINI_API_KEY in .env file or pass it directly."
            )
        self.client = genai.Client(api_key=key)
        self.model = "gemini-2.0-flash"

    def ask_with_context(
        self,
        question: str,
        context_chunks: List[str],
        source_name: str = "dokumen",
    ) -> str:
        """
        Answer a question using RAG - retrieves context chunks first,
        then asks Gemini to answer based only on that context.

        Args:
            question: User's question.
            context_chunks: List of relevant text chunks.
            source_name: Name of the source document.

        Returns:
            Generated answer text.
        """
        # Format context
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            context_text += f"\n[Dokumen {i+1}]\n{chunk.strip()}\n"

        # Build prompt
        prompt = f"""Anda adalah asisten yang membantu menjawab pertanyaan berdasarkan dokumen berikut.

=== DOKUMEN ({source_name}) ===
{context_text}

=== PERTANYAAN ===
{question}

=== INSTRUKSI ===
Jawab pertanyaan berdasarkan isi dokumen di atas.
Jika jawaban tidak ditemukan di dokumen, katakan bahwa Anda tidak tahu.
Jangan menambahkan informasi dari luar dokumen.
Jawab dalam Bahasa Indonesia.
Beri kutipan nomor dokumen yang relevan dalam jawaban Anda.
"""

        # Set system instruction via config
        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        )

        result = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return result.text


class GeminiEmbeddingService:
    """
    Service for generating embeddings using Gemini models.

    Supports two modes:
    1. Native PDF embedding: Send raw PDF bytes (≤6 pages).
    2. Text embedding: Send extracted/chunked text.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the embedding service.

        Args:
            api_key: Gemini API key. Falls back to Config if not provided.
        """
        key = api_key or Config.GEMINI_API_KEY
        if not key:
            raise ValueError(
                "Gemini API key is required. "
                "Set GEMINI_API_KEY in .env file or pass it directly."
            )
        self.client = genai.Client(api_key=key)
        self.model = Config.EMBEDDING_MODEL
        self.dimensionality = Config.EMBEDDING_DIMENSIONALITY

    def embed_pdf_bytes(self, pdf_bytes: bytes) -> List[float]:
        """
        Embed a PDF document using native multimodal support.

        This method sends the raw PDF bytes to gemini-embedding-2,
        which natively understands PDF documents.

        Args:
            pdf_bytes: Raw PDF file bytes.

        Returns:
            Embedding vector as a list of floats.

        Note:
            Limited to 1 PDF file per request, max 6 pages.
        """
        result = self.client.models.embed_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf",
                ),
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimensionality
            ),
        )
        return result.embeddings[0].values

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a text string.

        For best results with RAG-style retrieval, text is formatted
        with task prefix as recommended by Gemini embedding docs.

        Args:
            text: Text content to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        # Format with task instruction for better retrieval performance
        formatted_text = f"task: retrieval document | text: {text}"

        result = self.client.models.embed_content(
            model=self.model,
            contents=[formatted_text],
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimensionality
            ),
        )
        return result.embeddings[0].values

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        # Format each text with task instruction
        formatted_texts = [
            f"task: retrieval document | text: {t}" for t in texts
        ]

        result = self.client.models.embed_content(
            model=self.model,
            contents=formatted_texts,
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimensionality
            ),
        )
        return [e.values for e in result.embeddings]
