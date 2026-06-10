"""
Pydantic Models for RAG API

Request and response schemas for the chat API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Source(BaseModel):
    """A source document reference."""
    title: str = Field(..., description="Document title")
    path: str = Field(..., description="URL path to the document")
    module: str = Field(default="", description="Module name")
    snippet: str = Field(default="", description="Text snippet from the source")


class ChatRequest(BaseModel):
    """Request to the chat endpoint."""
    question: str = Field(..., description="User question", min_length=1, max_length=1000)
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")
    filter_module: Optional[str] = Field(default=None, description="Filter by module name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I create a ROS 2 node?",
                "conversation_id": None,
                "filter_module": None
            }
        }


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(default_factory=list, description="Source documents")
    conversation_id: str = Field(..., description="Conversation ID for follow-up")
    is_grounded: bool = Field(default=True, description="Whether the answer is grounded in sources")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "To create a ROS 2 node, you need to...",
                "sources": [
                    {
                        "title": "ROS 2 Architecture",
                        "path": "/docs/modules/ros2/architecture",
                        "module": "ros2",
                        "snippet": "Nodes are the fundamental building blocks..."
                    }
                ],
                "conversation_id": "abc123",
                "is_grounded": True,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Response from the health endpoint."""
    status: str = Field(..., description="Overall health status")
    qdrant: bool = Field(..., description="Qdrant connection status")
    openai: bool = Field(..., description="OpenAI API status")
    docs_count: int = Field(default=0, description="Number of documents in vector store")
    version: str = Field(..., description="API version")


class IngestRequest(BaseModel):
    """Request to trigger ingestion."""
    docs_path: str = Field(default="./docs", description="Path to documentation directory")
    force: bool = Field(default=False, description="Force re-ingestion of all documents")


class IngestResponse(BaseModel):
    """Response from ingestion endpoint."""
    success: bool = Field(..., description="Whether ingestion succeeded")
    total_files: int = Field(default=0, description="Total files found")
    processed_files: int = Field(default=0, description="Files successfully processed")
    total_chunks: int = Field(default=0, description="Total chunks created")
    errors: List[str] = Field(default_factory=list, description="Error messages if any")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
