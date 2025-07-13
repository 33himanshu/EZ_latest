import requests
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class OllamaEmbeddingClient:
    """Client for interacting with Ollama's embedding API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama embedding client.
        
        Args:
            base_url: Base URL of the Ollama API server
        """
        self.embed_url = f"{base_url.rstrip('/')}/api/embeddings"
        self.model_name = "nomic-embed-text"
        self.dimensions = 768  # Standard for nomic-embed-text
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def get_embeddings(self, texts: List[str], model: Optional[str] = None) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            model: Optional model name override
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
            
        model = model or self.model_name
        
        try:
            # Process in batches to avoid overwhelming the API
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = requests.post(
                    self.embed_url,
                    json={
                        "model": model,
                        "prompts": batch
                    },
                    timeout=60
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Handle both single and batch response formats
                if 'embeddings' in result:
                    batch_embeddings = result['embeddings']
                elif isinstance(result, list):
                    batch_embeddings = [item.get('embedding', []) for item in result]
                else:
                    batch_embeddings = [result.get('embedding', [])]
                
                all_embeddings.extend(batch_embeddings)
            
            return np.array(all_embeddings)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise
    
    def get_embedding(self, text: str, model: Optional[str] = None) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Input text
            model: Optional model name override
            
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        embeddings = self.get_embeddings([text], model)
        return embeddings[0] if embeddings.size > 0 else np.array([])


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""
    
    def __init__(self):
        self.cache = {}
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text if exists."""
        return self.cache.get(text)
    
    def set(self, text: str, embedding: np.ndarray) -> None:
        """Cache an embedding."""
        self.cache[text] = embedding
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
