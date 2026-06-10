"""
RAG Chatbot Core Library

Retrieval-Augmented Generation for the AI-Native Book.
"""

from .config import RAGConfig, get_config
from .embeddings import EmbeddingClient, get_embedding_client
from .vectorstore import VectorStore, get_vector_store
from .retriever import Retriever, get_retriever
from .generator import Generator, get_generator
from .ingestion import ingest_documents, ingest_file

__all__ = [
    "RAGConfig",
    "get_config",
    "EmbeddingClient",
    "get_embedding_client",
    "VectorStore",
    "get_vector_store",
    "Retriever",
    "get_retriever",
    "Generator",
    "get_generator",
    "ingest_documents",
    "ingest_file",
]

__version__ = "0.1.0"
