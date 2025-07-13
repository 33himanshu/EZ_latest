import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, asdict, field
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text with its metadata and embedding."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    chunk_id: str = ""
    document_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        if self.embedding is not None:
            result['embedding'] = self.embedding.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create from dictionary."""
        embedding = data.get('embedding')
        if embedding is not None:
            data['embedding'] = np.array(embedding, dtype=np.float32)
        return cls(**data)


class VectorStore:
    """Manages vector storage and similarity search using FAISS."""
    
    def __init__(self, persist_dir: str = "./vector_store", dimension: int = 768):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory to persist the index and metadata
            dimension: Dimensionality of the embeddings
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True, parents=True)
        self.dimension = dimension
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        self.chunks: List[DocumentChunk] = []
        self.documents: Dict[str, Dict] = {}  # document_id -> document_metadata
        
        # Load existing data if available
        self._load()
    
    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embeddings_client: Optional[Any] = None
    ) -> str:
        """
        Add a document to the vector store.
        
        Args:
            text: Document text
            metadata: Document metadata
            document_id: Optional document ID (auto-generated if None)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embeddings_client: Client for generating embeddings
            
        Returns:
            Document ID
        """
        from .reading_functions import TextChunker
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = self._generate_id(text)
        
        if metadata is None:
            metadata = {}
            
        # Store document metadata
        self.documents[document_id] = {
            'id': document_id,
            'metadata': metadata,
            'chunk_count': 0
        }
        
        # Chunk the document
        chunker = TextChunker()
        chunks = chunker.chunk_text(text, chunk_size, chunk_overlap)
        
        # Add chunks to the index
        chunk_objects = []
        for chunk_data in chunks:
            chunk = DocumentChunk(
                text=chunk_data['text'],
                metadata={
                    **metadata,
                    'start_char': chunk_data.get('start_char', 0),
                    'end_char': chunk_data.get('end_char', len(chunk_data['text']))
                },
                document_id=document_id,
                chunk_id=f"{document_id}_{len(self.chunks) + len(chunk_objects)}"
            )
            chunk_objects.append(chunk)
        
        # Generate embeddings if client is provided
        if embeddings_client and chunk_objects:
            texts = [chunk.text for chunk in chunk_objects]
            embeddings = embeddings_client.get_embeddings(texts)
            
            for chunk, embedding in zip(chunk_objects, embeddings):
                chunk.embedding = embedding
        
        # Add to index and store chunks
        self._add_chunks(chunk_objects)
        
        # Update document metadata
        self.documents[document_id]['chunk_count'] += len(chunk_objects)
        
        # Persist changes
        self._save()
        
        return document_id
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        embeddings_client: Optional[Any] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_conditions: Optional filter conditions on metadata
            embeddings_client: Client for generating query embedding
            
        Returns:
            List of (chunk, score) tuples
        """
        if not self.chunks:
            return []
        
        # Get query embedding
        if embeddings_client is None:
            raise ValueError("Embeddings client is required for search")
            
        query_embedding = embeddings_client.get_embedding(query)
        if query_embedding.size == 0:
            return []
        
        # Convert to numpy array and reshape for FAISS
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
        
        # Get the results
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
                
            chunk = self.chunks[idx]
            
            # Apply filters if any
            if filter_conditions and not self._matches_filters(chunk, filter_conditions):
                continue
                
            results.append((chunk, float(score)))
        
        return results
    
    def delete_document(self, document_id: str) -> None:
        """Delete a document and all its chunks."""
        if document_id not in self.documents:
            return
        
        # Find and remove all chunks for this document
        chunks_to_keep = []
        chunk_indices_to_keep = []
        
        for i, chunk in enumerate(self.chunks):
            if chunk.document_id != document_id:
                chunks_to_keep.append(chunk)
                chunk_indices_to_keep.append(i)
        
        # Rebuild the index with remaining chunks
        self.chunks = chunks_to_keep
        self._rebuild_index()
        
        # Remove document metadata
        del self.documents[document_id]
        
        # Persist changes
        self._save()
    
    def clear(self) -> None:
        """Clear the entire vector store."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.documents = {}
        self._save()
    
    def _add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks to the index."""
        if not chunks:
            return
        
        # Add to chunks list
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        
        # Add embeddings to FAISS index
        embeddings = []
        for chunk in chunks:
            if chunk.embedding is not None:
                # Normalize for cosine similarity
                embedding = chunk.embedding.astype('float32')
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            else:
                # Add zero vector for chunks without embeddings
                embeddings.append(np.zeros(self.dimension, dtype='float32'))
        
        if embeddings:
            embeddings = np.stack(embeddings)
            self.index.add(embeddings)
    
    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from chunks."""
        self.index = faiss.IndexFlatIP(self.dimension)
        
        embeddings = []
        for chunk in self.chunks:
            if chunk.embedding is not None:
                # Normalize for cosine similarity
                embedding = chunk.embedding.astype('float32')
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
        
        if embeddings:
            embeddings = np.stack(embeddings)
            self.index.add(embeddings)
    
    def _matches_filters(self, chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        """Check if a chunk matches all filter conditions."""
        for key, value in filters.items():
            if key not in chunk.metadata or chunk.metadata[key] != value:
                return False
        return True
    
    def _generate_id(self, text: str) -> str:
        """Generate a unique ID from text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _save(self) -> None:
        """Save the index and metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.index, str(self.persist_dir / "index.faiss"))
        
        # Save chunks metadata
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(self.persist_dir / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        # Save documents metadata
        with open(self.persist_dir / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def _load(self) -> None:
        """Load the index and metadata from disk."""
        try:
            # Load FAISS index
            if (self.persist_dir / "index.faiss").exists():
                self.index = faiss.read_index(str(self.persist_dir / "index.faiss"))
            
            # Load chunks metadata
            if (self.persist_dir / "chunks.json").exists():
                with open(self.persist_dir / "chunks.json", 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                    self.chunks = [DocumentChunk.from_dict(data) for data in chunks_data]
            
            # Load documents metadata
            if (self.persist_dir / "documents.json").exists():
                with open(self.persist_dir / "documents.json", 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                    
        except Exception as e:
            logger.warning(f"Error loading vector store: {e}")
            # Reset to empty state if loading fails
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks = []
            self.documents = {}
