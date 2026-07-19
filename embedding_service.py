"""
Embedding Service Module.

Handles communication with the Google Gemini Embedding API
and Groq API for text generation using the google-genai SDK.
"""

from typing import List, Optional
import requests

from google import genai
from google.genai import types

from config import Config


class GroqGenerationService:
    """
    Service for generating text using Groq API (for RAG / Q&A).
    Uses Groq's LLaMA models for text generation.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.GROQ_API_KEY
        if not self.api_key:
            raise ValueError(
                "Groq API key is required. "
                "Set GROQ_API_KEY in .env file or pass it directly."
            )
        self.model = Config.GROQ_MODEL
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def ask_with_context(
        self,
        question: str,
        context_chunks: List[str],
        source_name: str = "dokumen",
    ) -> str:
        """
        Answer a question using RAG - retrieves context chunks first,
        then asks Groq LLM to answer based only on that context.

        Args:
            question: User's question.
            context_chunks: List of relevant text chunks.
            source_name: Name of the source document.

        Returns:
            Generated answer text.
            
        Raises:
            RuntimeError: If API quota is exhausted or other API errors occur.
        """
        # Format context
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            context_text += f"\n[Dokumen {i+1}]\n{chunk.strip()}\n"

        # DEBUG: Log untuk melihat apakah 'Certifications' ada di context
        import os
        if os.getenv("DEBUG_LLM") == "1":
            print("\n" + "="*70)
            print("DEBUG: Context yang dikirim ke LLM")
            print("="*70)
            print(f"Total chunks: {len(context_chunks)}")
            for i, chunk in enumerate(context_chunks):
                has_cert = "Certifications" in chunk or "certificate" in chunk.lower()
                print(f"Chunk {i+1}: {len(chunk)} chars, Has 'Certifications': {has_cert}")
            print("="*70 + "\n")

        # Build prompt with bilingual instruction
        prompt = f"""Anda adalah asisten yang membantu menjawab pertanyaan berdasarkan dokumen berikut.

=== DOKUMEN ({source_name}) ===
{context_text}

=== PERTANYAAN ===
{question}

=== INSTRUKSI ===
Jawab pertanyaan berdasarkan isi dokumen di atas dengan TELITI dan LENGKAP.

PENTING: Dokumen mungkin dalam Bahasa Inggris atau Bahasa Indonesia. Pahami konten dalam bahasa apapun.
Mapping kata kunci bilingual:
- "magang" / "pengalaman" = "intern", "internship", "trainee", "work experience"
- "pendidikan" = "education"
- "keahlian" = "skills"
- "sertifikat" / "sertifikasi" = "certificate", "certification", "certifications", "awards"

CARA MENJAWAB:
1. Baca dokumen dengan cermat untuk mencari informasi yang relevan
2. Perhatikan bahwa istilah dalam pertanyaan mungkin berbeda bahasa dengan dokumen (gunakan mapping di atas)
3. Jika menemukan informasi yang relevan, jawab dengan lengkap dan detail
4. HANYA katakan "tidak tahu" jika benar-benar tidak ada informasi sama sekali setelah membaca dengan teliti

Jangan menambahkan informasi dari luar dokumen.
Jawab dalam Bahasa Indonesia.
Beri kutipan nomor dokumen yang relevan dalam jawaban Anda.
"""

        # Call Groq API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1024
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise RuntimeError(
                    "API Quota Groq habis! Silakan:\n"
                    "1. Tunggu beberapa menit dan coba lagi\n"
                    "2. Atau check usage di https://console.groq.com/\n"
                    "3. Atau upgrade ke paid tier"
                ) from e
            else:
                raise RuntimeError(f"Groq API error: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Groq API error: {str(e)}") from e


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

    def embed_text(self, text: str, task_type: str = "retrieval document") -> List[float]:
        """
        Embed a text string.

        For best results with RAG-style retrieval, text is formatted
        with task prefix as recommended by Gemini embedding docs.

        Args:
            text: Text content to embed.
            task_type: Task type for embedding. Use "retrieval document" for documents,
                      "retrieval query" for search queries.

        Returns:
            Embedding vector as a list of floats.
        """
        # Input validation
        if not text or not text.strip():
            raise ValueError("text cannot be empty or whitespace-only")
        
        # Format with task instruction for better retrieval performance
        formatted_text = f"task: {task_type} | text: {text}"

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
        # Input validation
        if not texts:
            raise ValueError("texts list cannot be empty")
        
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
