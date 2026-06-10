"""
Document Retriever

Implements semantic search to retrieve relevant passages.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .config import get_config, RAGConfig
from .embeddings import get_embedding_client, EmbeddingClient
from .vectorstore import get_vector_store, VectorStore, Document


@dataclass
class RetrievalResult:
    """Result from retrieval including documents and metadata."""
    documents: List[Document]
    query: str
    query_embedding: List[float]


class Retriever:
    """
    Semantic search retriever.
    
    Features:
    - Query embedding
    - Top-k similarity search
    - Score filtering
    - Metadata filtering
    - Deduplication
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            config: RAG configuration
            embedding_client: Embedding client to use
            vector_store: Vector store to use
        """
        self.config = config or get_config()
        self.embedding_client = embedding_client or get_embedding_client(self.config)
        self.vector_store = vector_store or get_vector_store(self.config)
        
        self.top_k = self.config.retrieval.top_k
        self.score_threshold = self.config.retrieval.score_threshold
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_module: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_module: Filter by module name
            
        Returns:
            RetrievalResult with matching documents
        """
        top_k = top_k or self.top_k
        score_threshold = score_threshold or self.score_threshold
        
        # Generate query embedding
        query_embedding = self.embedding_client.embed(query)
        
        # Build filter
        filter_dict = None
        if filter_module:
            filter_dict = {'module': filter_module}
        
        # Search vector store
        documents = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_dict=filter_dict
        )
        
        # Deduplicate similar passages
        documents = self._deduplicate(documents)
        
        return RetrievalResult(
            documents=documents,
            query=query,
            query_embedding=query_embedding
        )
    
    def _deduplicate(
        self,
        documents: List[Document],
        similarity_threshold: float = 0.95
    ) -> List[Document]:
        """
        Remove near-duplicate documents.
        
        Args:
            documents: List of documents
            similarity_threshold: Threshold for considering duplicates
            
        Returns:
            Deduplicated list
        """
        if len(documents) <= 1:
            return documents
        
        result = [documents[0]]
        
        for doc in documents[1:]:
            is_duplicate = False
            for existing in result:
                # Check text similarity (simple containment check)
                if doc.text in existing.text or existing.text in doc.text:
                    is_duplicate = True
                    break
                # Check if from same source and adjacent chunks
                if (doc.metadata.get('source_path') == existing.metadata.get('source_path') and
                    abs(doc.metadata.get('chunk_index', 0) - existing.metadata.get('chunk_index', 1)) <= 1):
                    # Keep the higher scoring one
                    if doc.score > existing.score:
                        result.remove(existing)
                        result.append(doc)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                result.append(doc)
        
        return result
    
    def format_context(
        self,
        documents: List[Document],
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Format retrieved documents as context for the LLM.
        
        Args:
            documents: List of retrieved documents
            max_tokens: Maximum tokens for context
            
        Returns:
            Formatted context string
        """
        max_tokens = max_tokens or self.config.retrieval.max_context_tokens
        
        context_parts = []
        current_tokens = 0
        
        for i, doc in enumerate(documents):
            # Estimate tokens (4 chars per token)
            doc_tokens = len(doc.text) // 4
            
            if current_tokens + doc_tokens > max_tokens:
                break
            
            # Format with source info
            source = doc.metadata.get('source_path', 'Unknown')
            title = doc.metadata.get('title', 'Untitled')
            
            context_parts.append(
                f"[Source {i+1}: {title}]\n{doc.text}"
            )
            current_tokens += doc_tokens
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_sources(self, documents: List[Document]) -> List[Dict[str, str]]:
        """
        Extract source information from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            List of source dictionaries
        """
        sources = []
        seen_paths = set()
        
        for doc in documents:
            path = doc.metadata.get('source_path', '')
            if path and path not in seen_paths:
                seen_paths.add(path)
                
                # Convert file path to docs URL
                doc_url = self._path_to_url(path)
                
                sources.append({
                    'title': doc.metadata.get('title', 'Untitled'),
                    'path': doc_url,
                    'module': doc.metadata.get('module', ''),
                    'snippet': doc.text[:200] + '...' if len(doc.text) > 200 else doc.text
                })
        
        return sources
    
    def _path_to_url(self, file_path: str) -> str:
        """Convert file path to documentation URL."""
        path = file_path.replace('\\', '/')

        # Remove extension
        for ext in ['.mdx', '.md']:
            if path.endswith(ext):
                path = path[:-len(ext)]

        # Find the 'docs/modules' or 'docs/' marker and take everything after 'docs/'
        # Docusaurus serves docs at /docs/<relative-path>
        # source_path is like: E:/ai-native-book/docs/modules/ros2/architecture
        # We want: /docs/modules/ros2/architecture
        markers = ['docs/modules/', 'docs/']
        for marker in markers:
            if marker in path:
                idx = path.index(marker)
                relative = path[idx + len('docs/'):]  # keep 'modules/...' part
                return f"/docs/{relative}"

        # Fallback: just use filename
        return f"/docs/{path.split('/')[-1]}"


# Global retriever instance
_retriever: Optional[Retriever] = None


def get_retriever(config: Optional[RAGConfig] = None) -> Retriever:
    """Get or create the global retriever."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(config)
    return _retriever
