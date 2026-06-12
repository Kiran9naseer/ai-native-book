"""
Embeddings Client

Generates embeddings using Google Gemini (native SDK) or OpenAI API or Hugging Face.
Priority: Gemini native > Hugging Face > OpenAI > Mock
"""

import time
from typing import List, Optional
from dataclasses import dataclass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import google.generativeai as genai_embed
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        from google import genai as _genai
        GEMINI_AVAILABLE = True
        genai_embed = None
    except ImportError:
        GEMINI_AVAILABLE = False
        genai_embed = None

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from .config import get_config, RAGConfig


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: List[float]
    token_count: int


class EmbeddingClient:
    """
    Client for generating embeddings.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or get_config()
        self._provider = "mock"
        self.model = None

        # 1. Try Gemini native SDK
        if GEMINI_AVAILABLE and genai_embed and self.config.gemini.api_key:
            try:
                genai_embed.configure(api_key=self.config.gemini.api_key)
                self._provider = "gemini"
                self.model = "models/embedding-001"
                print("[Embeddings] Using Gemini SDK (embedding-001, 768 dims)")
            except Exception as e:
                print(f"[Embeddings] Gemini SDK init failed: {e}")
                self._provider = "mock"

        # 2. Try Hugging Face
        elif HF_AVAILABLE and self.config.huggingface.api_key:
            self.hf_client = InferenceClient(token=self.config.huggingface.api_key, timeout=60)
            self._provider = "huggingface"
            self.model = "sentence-transformers/all-mpnet-base-v2"
            print(f"[Embeddings] Using Hugging Face ({self.model}, 768 dims)")

        # 3. Try OpenAI
        elif OPENAI_AVAILABLE and self.config.openai.api_key:
            self._openai_client = OpenAI(
                api_key=self.config.openai.api_key,
                base_url=self.config.openai.base_url
            )
            self._provider = "openai"
            self.model = self.config.openai.embedding_model
            print(f"[Embeddings] Using OpenAI ({self.model}, 1536 dims)")

        # 4. Fallback — mock
        else:
            print(f"[Embeddings] No API key found. Using mock embeddings ({self.config.qdrant.vector_size} dims)")
            self._provider = "mock"
            self.model = "mock"

        # Token encoder (for counting)
        if TIKTOKEN_AVAILABLE and self.model and self.model != "mock":
            try:
                self.encoder = tiktoken.encoding_for_model("text-embedding-ada-002")
            except Exception:
                self.encoder = None
        else:
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text) // 4

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self._provider == "gemini":
            return self._embed_gemini(text)
        elif self._provider == "huggingface":
            return self._embed_hf(text)
        elif self._provider == "openai":
            return self._embed_openai(text)
        return self._mock_embedding(text)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 50,
        show_progress: bool = False
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1

            if show_progress:
                print(f"Processing batch {batch_num}/{total_batches}...")

            embeddings = self._embed_batch_internal(batch)

            for text, embedding in zip(batch, embeddings):
                results.append(EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    token_count=self.count_tokens(text)
                ))

            # Rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.2)

        return results

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _embed_gemini(self, text: str) -> List[float]:
        """Embed using Gemini google.generativeai SDK."""
        try:
            result = genai_embed.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            return result["embedding"]
        except Exception as e:
            print(f"[Gemini embed error] {e}")
            return self._mock_embedding(text)

    def _embed_hf(self, text: str) -> List[float]:
        """Embed using Hugging Face."""
        try:
            res = self.hf_client.feature_extraction(text, model=self.model)
            return res[0] if isinstance(res[0], list) else res
        except Exception as e:
            print(f"[HF embed error] {e}")
            return self._mock_embedding(text)

    def _embed_openai(self, text: str) -> List[float]:
        """Embed using OpenAI API."""
        try:
            response = self._openai_client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[OpenAI embed error] {e}")
            return self._mock_embedding(text)

    def _embed_batch_internal(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        if self._provider == "gemini":
            return self._embed_batch_gemini(texts)
        elif self._provider == "huggingface":
            return self._embed_batch_hf(texts)
        elif self._provider == "openai":
            return self._embed_batch_openai(texts)
        return [self._mock_embedding(t) for t in texts]

    def _embed_batch_gemini(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """Batch embed using google.generativeai SDK."""
        results = []
        for text in texts:
            for attempt in range(max_retries):
                try:
                    result = genai_embed.embed_content(
                        model=self.model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    results.append(result["embedding"])
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        print(f"[Gemini] Retry {attempt+1} after {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        print(f"[Gemini] Embedding failed, using mock: {e}")
                        results.append(self._mock_embedding(text))
        return results

    def _embed_batch_hf(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        results = []
        for text in texts:
            for attempt in range(max_retries):
                try:
                    res = self._embed_hf(text)
                    results.append(res)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    else:
                        results.append(self._mock_embedding(text))
        return results

    def _embed_batch_openai(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """Batch embed using OpenAI API."""
        for attempt in range(max_retries):
            try:
                response = self._openai_client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[OpenAI] Retry {attempt+1} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"[OpenAI] Batch embedding failed (using mock): {e}")
                    return [self._mock_embedding(t) for t in texts]
        return [self._mock_embedding(t) for t in texts]

    def _mock_embedding(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding (correct dimensions)."""
        import hashlib
        dims = self.config.qdrant.vector_size  # 768 for Gemini, 1536 for OpenAI
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(dims):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            embedding.append(value)
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        return embedding


# Global client instance
_client: Optional[EmbeddingClient] = None


def get_embedding_client(config: Optional[RAGConfig] = None) -> EmbeddingClient:
    """Get or create the global embedding client."""
    global _client
    if _client is None:
        _client = EmbeddingClient(config)
    return _client
