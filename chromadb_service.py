"""
ChromaDB Service Module.

Handles connection to ChromaDB (local or cloud) and provides
methods for storing and retrieving embeddings.
"""

from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings

from config import Config


class ChromaDBService:
    """Service for managing ChromaDB operations."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        use_cloud: Optional[bool] = None
    ):
        """
        Initialize ChromaDB service.

        Args:
            collection_name: Name of the collection to use.
            use_cloud: Whether to use ChromaDB Cloud. If None, uses Config.USE_CHROMA_CLOUD.
        """
        self.collection_name = collection_name or Config.CHROMA_COLLECTION_NAME
        self.use_cloud = use_cloud if use_cloud is not None else Config.USE_CHROMA_CLOUD
        
        # Initialize client
        if self.use_cloud:
            self.client = self._init_cloud_client()
        else:
            self.client = self._init_local_client()
        
        # Get or create collection
        self.collection = None

    def _init_cloud_client(self) -> chromadb.CloudClient:
        """Initialize ChromaDB Cloud client."""
        if not Config.CHROMA_CLOUD_API_KEY:
            raise ValueError(
                "CHROMA_CLOUD_API_KEY is required for ChromaDB Cloud. "
                "Please set it in your .env file."
            )
        if not Config.CHROMA_CLOUD_TENANT:
            raise ValueError(
                "CHROMA_CLOUD_TENANT is required for ChromaDB Cloud. "
                "Please set it in your .env file."
            )
        if not Config.CHROMA_CLOUD_DATABASE:
            raise ValueError(
                "CHROMA_CLOUD_DATABASE is required for ChromaDB Cloud. "
                "Please set it in your .env file."
            )
        
        return chromadb.CloudClient(
            api_key=Config.CHROMA_CLOUD_API_KEY,
            tenant=Config.CHROMA_CLOUD_TENANT,
            database=Config.CHROMA_CLOUD_DATABASE
        )

    def _init_local_client(self) -> chromadb.PersistentClient:
        """Initialize local ChromaDB client."""
        return chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

    def get_or_create_collection(
        self,
        metadata: Optional[Dict[str, Any]] = None
    ) -> chromadb.Collection:
        """
        Get existing collection or create new one.

        Args:
            metadata: Optional metadata for the collection.

        Returns:
            ChromaDB collection object.
        """
        if self.collection is not None:
            return self.collection
        
        # Ensure metadata is not empty for ChromaDB Cloud
        if metadata is None or len(metadata) == 0:
            metadata = {"created_by": "pdf_embedding_starter"}
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=metadata
        )
        return self.collection

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add embeddings to the collection.

        Args:
            embeddings: List of embedding vectors.
            documents: List of document texts.
            ids: List of unique IDs for each document.
            metadatas: Optional list of metadata dictionaries.
        """
        if self.collection is None:
            self.get_or_create_collection()
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def query_embeddings(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query embeddings from the collection.

        Args:
            query_embeddings: List of query embedding vectors.
            n_results: Number of results to return per query.
            where: Optional metadata filter.
            where_document: Optional document content filter.

        Returns:
            Query results containing documents, metadatas, distances, etc.
        """
        if self.collection is None:
            self.get_or_create_collection()
        
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document
        )

    def delete_collection(self) -> None:
        """Delete the current collection."""
        if self.collection is not None:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None

    def count_documents(self) -> int:
        """Get the number of documents in the collection."""
        if self.collection is None:
            return 0
        return self.collection.count()

    def get_collection_metadata(self) -> Dict[str, Any]:
        """Get metadata for the current collection."""
        if self.collection is None:
            self.get_or_create_collection()
        return self.collection.metadata

    def list_collections(self) -> List[str]:
        """List all available collections."""
        collections = self.client.list_collections()
        return [col.name for col in collections]

    def delete_documents_by_ids(self, ids: List[str]) -> None:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete.
        """
        if self.collection is None:
            self.get_or_create_collection()
        
        self.collection.delete(ids=ids)

    def update_documents(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Update existing documents in the collection.

        Args:
            ids: List of document IDs to update.
            embeddings: Optional new embeddings.
            documents: Optional new document texts.
            metadatas: Optional new metadata.
        """
        if self.collection is None:
            self.get_or_create_collection()
        
        self.collection.update(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
