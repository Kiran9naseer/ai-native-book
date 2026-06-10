"""
Content Ingestion Module

Processes MDX documentation files and ingests them into the vector store.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator
from dataclasses import dataclass

try:
    import frontmatter
    FRONTMATTER_AVAILABLE = True
except ImportError:
    FRONTMATTER_AVAILABLE = False

from .config import get_config, RAGConfig
from .embeddings import get_embedding_client, EmbeddingClient
from .vectorstore import get_vector_store, VectorStore, Document
import uuid


@dataclass
class Chunk:
    """A chunk of document content."""
    text: str
    metadata: Dict[str, Any]


class MDXParser:
    """
    Parser for MDX documentation files.
    
    Extracts content and metadata from MDX files without regex.
    """
    
    def __init__(self):
        pass  # No regex patterns needed
    
    def parse_file(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """
        Parse an MDX file and extract content and metadata.
        
        Args:
            file_path: Path to the MDX file
            
        Returns:
            Tuple of (content, metadata)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        # Parse frontmatter
        metadata = {}
        content = raw_content
        
        if FRONTMATTER_AVAILABLE:
            try:
                parsed = frontmatter.loads(raw_content)
                content = parsed.content
                metadata = dict(parsed.metadata)
            except Exception:
                pass
        else:
            # Simple frontmatter extraction without regex
            if raw_content.startswith('---'):
                parts = raw_content.split('---', 2)
                if len(parts) >= 3:
                    content = parts[2].strip()
        
        # Add file metadata
        metadata['source_path'] = str(file_path)
        metadata['filename'] = file_path.name
        
        # Extract module from path
        parts = file_path.parts
        if 'modules' in parts:
            idx = parts.index('modules')
            if idx + 1 < len(parts):
                metadata['module'] = parts[idx + 1]
        
        return content, metadata
    
    def clean_content(self, content: str) -> str:
        """
        Clean MDX content for embedding without regex.
        
        Args:
            content: Raw MDX content
            
        Returns:
            Cleaned plain text
        """
        text = content
        
        # Remove code blocks (find ``` pairs and replace)
        while '```' in text:
            start = text.find('```')
            if start == -1:
                break
            end = text.find('```', start + 3)
            if end == -1:
                break
            text = text[:start] + '\n[Code example]\n' + text[end + 3:]
        
        # Remove HTML-like JSX tags (simple approach)
        result = []
        in_tag = False
        for char in text:
            if char == '<':
                in_tag = True
            elif char == '>':
                in_tag = False
            elif not in_tag:
                result.append(char)
        text = ''.join(result)
        
        # Simplify links: [text](url) -> text
        while '[' in text and '](' in text:
            start = text.find('[')
            mid = text.find('](', start)
            end = text.find(')', mid)
            if start == -1 or mid == -1 or end == -1:
                break
            link_text = text[start + 1:mid]
            text = text[:start] + link_text + text[end + 1:]
        
        # Clean up excess whitespace
        lines = text.split('\n')
        cleaned_lines = []
        blank_count = 0
        for line in lines:
            if line.strip() == '':
                blank_count += 1
                if blank_count <= 2:
                    cleaned_lines.append('')
            else:
                blank_count = 0
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def extract_title(self, content: str, metadata: Dict) -> str:
        """Extract title from content or metadata without regex."""
        # Try metadata first
        if 'title' in metadata:
            return metadata['title']
        
        # Try first heading (look for lines starting with #)
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                # Remove # symbols and whitespace
                title = line.lstrip('#').strip()
                if title:
                    return title
        
        return metadata.get('filename', 'Untitled')


class Chunker:
    """
    Splits content into chunks for embedding.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or get_config()
        self.chunk_size = self.config.chunking.chunk_size
        self.chunk_overlap = self.config.chunking.chunk_overlap
        self.min_chunk_size = self.config.chunking.min_chunk_size
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []
        
        # Split by paragraphs (double newlines) without regex
        paragraphs = []
        current_para = []
        lines = text.split('\n')
        for line in lines:
            if line.strip() == '':
                if current_para:
                    paragraphs.append('\n'.join(current_para))
                    current_para = []
            else:
                current_para.append(line)
        if current_para:
            paragraphs.append('\n'.join(current_para))
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Rough token estimate: 4 chars per token
            para_size = len(para) // 4
            
            if current_size + para_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                if len(chunk_text) // 4 >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={**metadata, 'chunk_index': len(chunks)}
                    ))
                
                # Keep overlap
                overlap_size = 0
                overlap_chunks = []
                for c in reversed(current_chunk):
                    c_size = len(c) // 4
                    if overlap_size + c_size <= self.chunk_overlap:
                        overlap_chunks.insert(0, c)
                        overlap_size += c_size
                    else:
                        break
                
                current_chunk = overlap_chunks
                current_size = overlap_size
            
            current_chunk.append(para)
            current_size += para_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) // 4 >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, 'chunk_index': len(chunks)}
                ))
        
        return chunks


def find_mdx_files(docs_path: Path) -> Generator[Path, None, None]:
    """Find all MDX and MD files in a directory."""
    for ext in ['*.mdx', '*.md']:
        yield from docs_path.rglob(ext)


def ingest_file(
    file_path: Path,
    embedding_client: Optional[EmbeddingClient] = None,
    vector_store: Optional[VectorStore] = None,
    config: Optional[RAGConfig] = None
) -> int:
    """
    Ingest a single file into the vector store.
    
    Args:
        file_path: Path to the MDX file
        embedding_client: Embedding client to use
        vector_store: Vector store to use
        config: RAG configuration
        
    Returns:
        Number of chunks ingested
    """
    config = config or get_config()
    embedding_client = embedding_client or get_embedding_client(config)
    vector_store = vector_store or get_vector_store(config)
    
    parser = MDXParser()
    chunker = Chunker(config)
    
    # Parse file
    content, metadata = parser.parse_file(file_path)
    
    # Clean and extract title
    clean_content = parser.clean_content(content)
    title = parser.extract_title(content, metadata)
    metadata['title'] = title
    
    # Chunk content
    chunks = chunker.chunk_text(clean_content, metadata)
    
    if not chunks:
        return 0
    
    # Generate embeddings
    texts = [chunk.text for chunk in chunks]
    embedding_results = embedding_client.embed_batch(texts)
    
    # Create documents
    documents = []
    for chunk, emb_result in zip(chunks, embedding_results):
        doc_id = str(uuid.uuid4())
        documents.append(Document(
            id=doc_id,
            text=chunk.text,
            embedding=emb_result.embedding,
            metadata=chunk.metadata,
        ))
    
    # Upsert to vector store
    count = vector_store.upsert(documents)
    return count


def ingest_documents(
    docs_path: str,
    config: Optional[RAGConfig] = None,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Ingest all documentation files into the vector store.
    
    Args:
        docs_path: Path to the docs directory
        config: RAG configuration
        show_progress: Show progress output
        
    Returns:
        Dictionary with ingestion stats
    """
    config = config or get_config()
    embedding_client = get_embedding_client(config)
    vector_store = get_vector_store(config)
    
    docs_dir = Path(docs_path)
    if not docs_dir.exists():
        raise ValueError(f"Docs path does not exist: {docs_path}")
    
    # Find all files
    files = list(find_mdx_files(docs_dir))
    
    if show_progress:
        print(f"Found {len(files)} documentation files")
    
    stats = {
        'total_files': len(files),
        'processed_files': 0,
        'total_chunks': 0,
        'errors': []
    }
    
    for i, file_path in enumerate(files):
        if show_progress:
            print(f"[{i+1}/{len(files)}] Processing: {file_path.name}")
        
        try:
            count = ingest_file(file_path, embedding_client, vector_store, config)
            stats['processed_files'] += 1
            stats['total_chunks'] += count
        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            stats['errors'].append(error_msg)
            if show_progress:
                print(f"  Error: {e}")
    
    if show_progress:
        print(f"\nIngestion complete!")
        print(f"  Files processed: {stats['processed_files']}/{stats['total_files']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        if stats['errors']:
            print(f"  Errors: {len(stats['errors'])}")
    
    return stats
