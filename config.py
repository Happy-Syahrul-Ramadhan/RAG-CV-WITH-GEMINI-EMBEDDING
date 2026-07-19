"""
Configuration module for PDF Embedding Starter.

Loads environment variables and provides configuration for
Gemini embedding, Groq generation, ChromaDB, and PDF processing.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class."""

    # Google Gemini API (for embeddings only)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Groq API (for LLM generation)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Embedding model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
    EMBEDDING_DIMENSIONALITY: int = int(
        os.getenv("EMBEDDING_DIMENSIONALITY", "768")
    )
    
    # Groq model
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ChromaDB settings
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "pdf_embeddings"
    )
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", "./chroma_db"
    )
    
    # ChromaDB Cloud settings
    CHROMA_CLOUD_API_KEY: str = os.getenv("CHROMA_CLOUD_API_KEY", "")
    CHROMA_CLOUD_TENANT: str = os.getenv("CHROMA_CLOUD_TENANT", "")
    CHROMA_CLOUD_DATABASE: str = os.getenv("CHROMA_CLOUD_DATABASE", "")
    USE_CHROMA_CLOUD: bool = os.getenv("USE_CHROMA_CLOUD", "false").lower() == "true"

    # PDF chunking settings (for long PDFs > 6 pages)
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    @classmethod
    def validate(cls) -> bool:
        """Validate that required config is present."""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set! "
                "Please set it in your .env file or environment variables.\n"
                "Get your API key at: https://aistudio.google.com/app/apikey"
            )
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set! "
                "Please set it in your .env file or environment variables.\n"
                "Get your API key at: https://console.groq.com/keys"
            )
        return True
