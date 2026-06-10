# Hugging Face Spaces Dockerfile for RAG Chatbot
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY app.py .
COPY docs/ ./docs/

# Copy qdrant_db if it exists (for pre-built vector store)
COPY qdrant_db/ ./qdrant_db/

# Ensure proper permissions
RUN chown -R user:user /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV QDRANT_URL=./qdrant_db

# Expose port for HF Spaces
EXPOSE 7860

USER user

# Run the application
CMD ["python", "app.py"]
