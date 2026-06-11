"""
FastAPI Application for RAG Chatbot

Main application entry point with middleware and configuration.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from src.rag.api.routes import router
from src.rag.api.models import ErrorResponse


# Create FastAPI app
app = FastAPI(
    title="AI-Native Book RAG Chatbot",
    description="Retrieval-Augmented Generation chatbot for the AI-Native Book on Physical AI & Humanoid Robotics",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Docusaurus dev server
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "https://kirannaseer-ai-native-book.hf.space",  # HF Spaces
        "https://huggingface.co",
        "https://*.hf.space",
        "https://ai-native-book.vercel.app",  # Vercel frontend
        "https://*.vercel.app",  # Vercel preview deployments
        "*",  # Allow all origins for production API
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add response timing header."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.3f}"
    return response


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc) if app.debug else "An unexpected error occurred"
        ).model_dump()
    )


# Include routers
app.include_router(router, prefix="/api")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI-Native Book RAG Chatbot",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/health",
        "chat": "/api/chat"
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    from ..config import get_config
    from ..vectorstore import get_vector_store
    
    print("🚀 Starting RAG Chatbot API...", flush=True)
    
    # Verify configuration
    config = get_config()
    if not config.gemini.api_key and not config.openai.api_key and not config.huggingface.api_key:
        print("⚠️  Warning: Neither GEMINI_API_KEY, OPENAI_API_KEY nor HF_TOKEN set. Using mock responses.", flush=True)
    else:
        print("🔑 API Key detected! Real AI responses enabled.", flush=True)
    
    # Initialize vector store
    try:
        store = get_vector_store()
        count = store.count()
        print(f"📚 Connected to Qdrant. Documents: {count}", flush=True)
    except Exception as e:
        print(f"⚠️  Warning: Could not connect to Qdrant: {e}", flush=True)
    
    print("✅ RAG Chatbot API ready!", flush=True)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 Shutting down RAG Chatbot API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
