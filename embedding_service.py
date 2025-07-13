import requests
from typing import List, Optional
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating text embeddings using Ollama."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        """Initialize the embedding service.
        
        Args:
            base_url: Base URL of the Ollama API server
            model: Name of the embedding model to use
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.embed_url = f"{self.base_url}/api/embeddings"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        try:
            # Process in batches to avoid overwhelming the API
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = requests.post(
                    self.embed_url,
                    json={
                        "model": self.model,
                        "prompt": batch[0]  # Ollama expects a single prompt for embeddings
                    },
                    timeout=60
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Handle response format
                if 'embedding' in result:
                    all_embeddings.append(result['embedding'])
                else:
                    logger.warning(f"Unexpected response format: {result}")
                    all_embeddings.append([0.0] * 768)  # Fallback zero vector
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Get embeddings for a batch of texts with progress tracking.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
