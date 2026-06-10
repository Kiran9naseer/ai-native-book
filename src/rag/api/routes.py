"""
API Routes for RAG Chatbot

Defines the REST API endpoints.
"""

import uuid
from typing import Dict
from fastapi import APIRouter, HTTPException, BackgroundTasks

from src.rag.api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    Source,
)
from src.rag.config import get_config
from src.rag.retriever import get_retriever
from src.rag.generator import get_generator
from src.rag.vectorstore import get_vector_store
from src.rag.ingestion import ingest_documents


router = APIRouter()

# In-memory conversation storage (use Redis/DB in production)
conversations: Dict[str, list] = {}


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Ask a question and get an answer based on the documentation.
    
    The answer is grounded in the AI-Native Book content and includes
    source citations.
    """
    try:
        # Get or create conversation
        conv_id = request.conversation_id or str(uuid.uuid4())
        history = conversations.get(conv_id, [])
        
        # Get retriever and generator
        retriever = get_retriever()
        generator = get_generator()
        
        # Retrieve relevant documents
        retrieval_result = retriever.retrieve(
            query=request.question,
            filter_module=request.filter_module
        )
        
        # Generate answer
        result = generator.generate(
            question=request.question,
            retrieval_result=retrieval_result,
            conversation_history=history
        )
        
        # Update conversation history
        history.append({"role": "user", "content": request.question})
        history.append({"role": "assistant", "content": result.answer})
        conversations[conv_id] = history[-10:]  # Keep last 10 messages
        
        # Convert sources to response format
        sources = [
            Source(
                title=s.get('title', 'Untitled'),
                path=s.get('path', ''),
                module=s.get('module', ''),
                snippet=s.get('snippet', '')
            )
            for s in result.sources
        ]
        
        return ChatResponse(
            answer=result.answer,
            sources=sources,
            conversation_id=conv_id,
            is_grounded=result.is_grounded
        )
        
    except Exception as e:
        import traceback
        print("Error in chat endpoint:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Check the health of the RAG system.
    
    Returns the status of Qdrant connection, OpenAI API, and document count.
    """
    from src.rag import __version__
    
    config = get_config()
    vector_store = get_vector_store()
    
    # Check Qdrant
    qdrant_ok = vector_store.health_check()
    
    # Check OpenAI (just verify API key is set)
    openai_ok = bool(config.openai.api_key)
    
    # Get document count
    docs_count = vector_store.count() if qdrant_ok else 0
    
    # Determine overall status
    if qdrant_ok and openai_ok:
        status = "healthy"
    elif qdrant_ok or openai_ok:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        qdrant=qdrant_ok,
        openai=openai_ok,
        docs_count=docs_count,
        version=__version__
    )


@router.post("/ingest", response_model=IngestResponse, tags=["Admin"])
async def trigger_ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks
) -> IngestResponse:
    """
    Trigger ingestion of documentation.
    
    This runs in the background and may take several minutes.
    Requires admin authentication in production.
    """
    try:
        # Run ingestion in background for large document sets
        # For now, run synchronously for simplicity
        stats = ingest_documents(request.docs_path, show_progress=False)
        
        return IngestResponse(
            success=len(stats.get('errors', [])) == 0,
            total_files=stats.get('total_files', 0),
            processed_files=stats.get('processed_files', 0),
            total_chunks=stats.get('total_chunks', 0),
            errors=stats.get('errors', [])
        )
        
    except Exception as e:
        return IngestResponse(
            success=False,
            errors=[str(e)]
        )


@router.delete("/conversations/{conversation_id}", tags=["Chat"])
async def delete_conversation(conversation_id: str) -> Dict[str, str]:
    """Delete a conversation from memory."""
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"status": "deleted"}
    return {"status": "not_found"}
