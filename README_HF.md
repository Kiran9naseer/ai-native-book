---
title: AI-Native Book RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AI-Native Book RAG Chatbot

RAG-powered chatbot API for the AI-Native Book on Physical AI & Humanoid Robotics.

## API Endpoints

- `GET /` — API info
- `GET /api/health` — Health check
- `POST /api/chat` — Ask a question
- `POST /api/ingest` — Ingest documents
- `GET /docs` — Swagger UI

## Environment Variables (Secrets)

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face API token |
| `HF_MODEL` | Model for chat generation |
| `QDRANT_URL` | Qdrant storage path |
| `QDRANT_COLLECTION` | Collection name |
| `VECTOR_SIZE` | Vector dimensions (768) |
