"""
Hugging Face Spaces Entry Point

Runs the FastAPI RAG Chatbot backend for production deployment.
"""

import uvicorn
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from src.rag.api.main import app

if __name__ == "__main__":
    # HF Spaces requires port 7860
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )
