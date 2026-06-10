"""
Qdrant Vector Store

Manages vector storage and retrieval using Qdrant.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from .config import get_config, RAGConfig


@dataclass
class Document:
    """Represents a document chunk in the vector store."""
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    score: float = 0.0


class VectorStore:
    """
    Qdrant vector database interface.
    
    Features:
    - Collection management
    - Vector upsert with metadata
    - Similarity search with filtering
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the vector store.
        
        Args:
            config: RAG configuration (uses global config if not provided)
        """
        self.config = config or get_config()
        self.collection_name = self.config.qdrant.collection_name
        self.vector_size = self.config.qdrant.vector_size
        
        if QDRANT_AVAILABLE:
            url = self.config.qdrant.url
            if url.startswith("http://") or url.startswith("https://"):
                self.client = QdrantClient(
                    url=url,
                    api_key=self.config.qdrant.api_key,
                )
            else:
                 # Local path or :memory:
                 self.client = QdrantClient(path=url)
        else:
            self.client = None
            self._mock_store: Dict[str, Document] = {}
        
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the collection exists."""
        if not self.client:
            return
        
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.create_collection()
        except Exception as e:
            print(f"Error checking collection: {e}")
    
    def create_collection(self):
        """Create the vector collection."""
        if not self.client:
            print("[Mock] Creating collection")
            return
        
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=distance_map.get(self.config.qdrant.distance, Distance.COSINE),
                ),
            )
            print(f"Created collection: {self.collection_name}")
        except Exception as e:
            print(f"Error creating collection: {e}")
    
    def delete_collection(self):
        """Delete the vector collection."""
        if not self.client:
            self._mock_store.clear()
            return
        
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def upsert(self, documents: List[Document]) -> int:
        """
        Insert or update documents in the vector store.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Number of documents upserted
        """
        if not documents:
            return 0
        
        if not self.client:
            # Mock storage
            for doc in documents:
                self._mock_store[doc.id] = doc
            return len(documents)
        
        points = [
            PointStruct(
                id=doc.id,
                vector=doc.embedding,
                payload={
                    "text": doc.text,
                    **doc.metadata
                }
            )
            for doc in documents
        ]
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            return len(documents)
        except Exception as e:
            print(f"Error upserting documents: {e}")
            return 0
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_dict: Optional metadata filters
            
        Returns:
            List of matching Documents
        """
        if not self.client:
            # Mock search: return all with fake scores
            results = list(self._mock_store.values())[:top_k]
            for doc in results:
                doc.score = 0.85  # Fake score
            return results
        
        # Build filter if provided
        query_filter = None
        if filter_dict:
            conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
                for key, value in filter_dict.items()
            ]
            query_filter = Filter(must=conditions) if conditions else None
        
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
            ).points
            
            return [
                Document(
                    id=str(result.id),
                    text=result.payload.get("text", ""),
                    embedding=[],  # Don't return full embedding
                    metadata={k: v for k, v in result.payload.items() if k != "text"},
                    score=result.score,
                )
                for result in results
            ]
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def count(self) -> int:
        """Get the number of documents in the collection."""
        if not self.client:
            return len(self._mock_store)
        
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            print(f"Error getting count: {e}")
            return 0
    
    def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        if not self.client:
            return True  # Mock is always healthy
        
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False


# Global store instance
_store: Optional[VectorStore] = None


def get_vector_store(config: Optional[RAGConfig] = None) -> VectorStore:
    """Get or create the global vector store."""
    global _store
    if _store is None:
        _store = VectorStore(config)
    return _store
