import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector storage and similarity search using FAISS."""
    
    def __init__(self, persist_dir: str = "./vector_store", dimension: int = 768):
        """Initialize the vector store.
        
        Args:
            persist_dir: Directory to store the vector index and metadata
            dimension: Dimensionality of the embeddings
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True, parents=True)
        self.dimension = dimension
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        self.documents = {}  # doc_id -> document metadata
        self.chunks = []     # List of (doc_id, chunk_metadata)
        
        # Load existing data if available
        self._load()
    
    def add_document(
        self, 
        doc_id: str, 
        chunks: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a document and its chunks to the vector store.
        
        Args:
            doc_id: Unique document identifier
            chunks: List of chunk dictionaries with 'text' and 'embedding' keys
            metadata: Optional document metadata
        """
        if not chunks:
            return
            
        # Store document metadata
        self.documents[doc_id] = {
            'id': doc_id,
            'metadata': metadata or {},
            'num_chunks': len(chunks)
        }
        
        # Add chunks
        start_idx = len(self.chunks)
        
        for chunk in chunks:
            self.chunks.append((doc_id, {
                'text': chunk.get('text', ''),
                'metadata': chunk.get('metadata', {})
            }))
        
        # Add embeddings to FAISS index
        embeddings = np.array([chunk['embedding'] for chunk in chunks], dtype='float32')
        if embeddings.size > 0:
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        
        # Save changes
        self._save()
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 5,
        doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            doc_id: Optional document ID to filter results
            
        Returns:
            List of matching chunks with metadata and scores
        """
        if not self.chunks or len(query_embedding) != self.dimension:
            return []
        
        # Prepare query
        query = np.array([query_embedding], dtype='float32')
        faiss.normalize_L2(query)  # Normalize for cosine similarity
        
        # Search
        distances, indices = self.index.search(query, min(k, len(self.chunks)))
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue
                
            doc_id_match, chunk = self.chunks[idx]
            
            # Filter by document ID if specified
            if doc_id is not None and doc_id_match != doc_id:
                continue
                
            results.append({
                'text': chunk['text'],
                'score': float(score),
                'document_id': doc_id_match,
                'metadata': chunk['metadata']
            })
        
        return results
    
    def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its chunks."""
        if doc_id not in self.documents:
            return
        
        # Find chunks to keep
        chunks_to_keep = []
        chunk_indices_to_keep = []
        
        for i, (doc_id_match, _) in enumerate(self.chunks):
            if doc_id_match != doc_id:
                chunks_to_keep.append((doc_id_match, self.chunks[i][1]))
                chunk_indices_to_keep.append(i)
        
        # Rebuild index with remaining chunks
        self.chunks = chunks_to_keep
        self._rebuild_index(chunk_indices_to_keep)
        
        # Remove document metadata
        del self.documents[doc_id]
        
        # Save changes
        self._save()
    
    def clear(self) -> None:
        """Clear the vector store."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = {}
        self.chunks = []
        self._save()
    
    def _rebuild_index(self, chunk_indices: List[int]) -> None:
        """Rebuild the FAISS index with the specified chunk indices."""
        if not chunk_indices:
            self.index = faiss.IndexFlatIP(self.dimension)
            return
        
        # Extract embeddings for the specified chunks
        embeddings = []
        for i in chunk_indices:
            if i < len(self.chunks):
                doc_id, chunk = self.chunks[i]
                if 'embedding' in chunk:
                    embeddings.append(chunk['embedding'])
        
        if not embeddings:
            self.index = faiss.IndexFlatIP(self.dimension)
            return
        
        # Convert to numpy array and normalize
        embeddings = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings)
        
        # Create new index and add embeddings
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
    
    def _save(self) -> None:
        """Save the vector store to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.persist_dir / "index.faiss"))
            
            # Save metadata
            data = {
                'documents': self.documents,
                'chunks': [
                    {
                        'doc_id': doc_id,
                        'text': chunk['text'],
                        'metadata': chunk.get('metadata', {})
                    }
                    for doc_id, chunk in self.chunks
                ]
            }
            
            with open(self.persist_dir / "metadata.json", 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
    
    def _load(self) -> None:
        """Load the vector store from disk."""
        try:
            # Load FAISS index
            index_path = self.persist_dir / "index.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            metadata_path = self.persist_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', {})
                    self.chunks = [
                        (item['doc_id'], {
                            'text': item['text'],
                            'metadata': item.get('metadata', {})
                        })
                        for item in data.get('chunks', [])
                    ]
                    
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            # Reset to empty state
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = {}
            self.chunks = []
