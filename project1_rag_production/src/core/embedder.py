"""
Embedder Module
Handles text embedding using sentence transformers
"""

from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Manages embedding models for text vectorization
    
    Supports any SentenceTransformer model from HuggingFace
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ):
        """
        Initialize Embedder
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', 'mps')
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to normalize embeddings (recommended for cosine similarity)
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        # Load model
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.max_seq_length = self.model.max_seq_length
        
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size (uses default if None)
            show_progress_bar: Show progress during encoding
            convert_to_numpy: Convert to numpy array
            
        Returns:
            Embeddings as numpy array
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=self.normalize_embeddings
        )
        
        return embeddings
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries (alias for encode, but semantically clear)"""
        return self.encode(queries, show_progress_bar=False)
    
    def encode_corpus(self, corpus: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode corpus documents (with progress bar by default)"""
        return self.encode(corpus, show_progress_bar=show_progress)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'max_seq_length': self.max_seq_length,
            'device': self.device,
            'normalize_embeddings': self.normalize_embeddings
        }



