"""
ChromaDB Service Module.

Handles storing embeddings in ChromaDB and performing
semantic search queries.
"""

from typing import List, Optional, Dict, Any

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from chromadb.config import Settings

from config import Config
from embedding_service import GeminiEmbeddingService


class GeminiChromaEmbeddingFunction(EmbeddingFunction):
    """
    Custom ChromaDB embedding function that uses Gemini.

    This allows ChromaDB to automatically compute embeddings
    when adding documents or running queries.
    """

    def __init__(self, embedding_service: GeminiEmbeddingService):
        self.service = embedding_service

    def __call__(self, input: Documents) -> Embeddings:
        return self.service.embed_texts(list(input))


class ChromaService:
    """
    Service for interacting with ChromaDB.

    Handles collection creation, document indexing, and
    similarity search.
    """

    def __init__(
        self,
        embedding_service: Optional[GeminiEmbeddingService] = None,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize ChromaDB service.

        Args:
            embedding_service: Gemini embedding service instance.
            persist_dir: Directory for persistent ChromaDB storage.
            collection_name: Name of the ChromaDB collection.
        """
        self.persist_dir = persist_dir or Config.CHROMA_PERSIST_DIR
        self.collection_name = (
            collection_name or Config.CHROMA_COLLECTION_NAME
        )

        # Initialize embedding function
        if embedding_service is None:
            embedding_service = GeminiEmbeddingService()
        self.embedding_function = GeminiChromaEmbeddingFunction(
            embedding_service
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents with embeddings to ChromaDB.

        Args:
            documents: List of text documents to add.
            metadatas: Optional metadata for each document.
            ids: Optional IDs for each document. Auto-generated if None.

        Returns:
            List of document IDs.
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        return ids

    def add_pdf_chunks(
        self,
        chunks: List[str],
        source_pdf: str,
        page_numbers: Optional[List[int]] = None,
    ) -> List[str]:
        """
        Add PDF text chunks to ChromaDB with source metadata.

        Args:
            chunks: List of text chunks from a PDF.
            source_pdf: Filename of the source PDF.
            page_numbers: Optional page numbers for each chunk.

        Returns:
            List of document IDs.
        """
        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = {
                "source": source_pdf,
                "chunk_index": i,
                "char_count": len(chunk),
            }
            if page_numbers and i < len(page_numbers):
                meta["page"] = page_numbers[i]
            metadatas.append(meta)

        return self.add_documents(
            documents=chunks,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar documents using semantic search.

        Args:
            query: Natural language query text.
            n_results: Number of results to return.
            where: Optional filter conditions.

        Returns:
            Dictionary with documents, metadatas, distances, and ids.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0],
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_dir,
        }

    def delete_collection(self):
        """Delete the current collection."""
        self.client.delete_collection(self.collection_name)

    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        return [c.name for c in self.client.list_collections()]
