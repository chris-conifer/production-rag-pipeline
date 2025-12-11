"""
Retriever Module
Vector search using FAISS
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from pathlib import Path


class Retriever:
    """
    Manages vector search using FAISS
    
    Supports multiple index types:
    - IndexFlatL2: Exact search (best quality)
    - IndexIVFFlat: Approximate search (faster)
    - IndexHNSWFlat: Graph-based search (good balance)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "IndexFlatL2",
        similarity_metric: str = "cosine",
        nlist: int = 100,
        nprobe: int = 10
    ):
        """
        Initialize Retriever
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: FAISS index type
            similarity_metric: 'cosine', 'l2', or 'dot_product'
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search in IVF
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.similarity_metric = similarity_metric
        self.nlist = nlist
        self.nprobe = nprobe
        
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
    def build_index(
        self,
        embeddings: np.ndarray,
        chunks: List[str] = None,
        chunk_metadata: List[Dict] = None
    ):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings (N, D)
            chunks: List of chunk texts
            chunk_metadata: List of chunk metadata dicts
        """
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Normalize if using cosine similarity
        if self.similarity_metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Create index based on type
        if self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe
        elif self.index_type == "IndexHNSWFlat":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks = chunks if chunks is not None else []
        self.chunk_metadata = chunk_metadata if chunk_metadata is not None else []
        
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding (1, D) or (D,)
            top_k: Number of results to return
            return_scores: Include similarity scores
            
        Returns:
            List of search results with metadata
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Normalize if using cosine similarity
        if self.similarity_metric == "cosine":
            faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:  # FAISS returns -1 for not found
                continue
                
            result = {
                'index': int(idx),
                'distance': float(dist),
            }
            
            if return_scores:
                # Convert distance to similarity score
                if self.similarity_metric == "cosine":
                    result['score'] = 1.0 - dist  # Cosine similarity
                else:
                    result['score'] = 1.0 / (1.0 + dist)  # General score
            
            if self.chunks and idx < len(self.chunks):
                result['text'] = self.chunks[idx]
            
            if self.chunk_metadata and idx < len(self.chunk_metadata):
                result['metadata'] = self.chunk_metadata[idx]
            
            results.append(result)
        
        return results
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[List[Dict]]:
        """
        Batch search for multiple queries
        
        Args:
            query_embeddings: Query embeddings (N, D)
            top_k: Number of results per query
            
        Returns:
            List of result lists
        """
        query_embeddings = query_embeddings.astype('float32')
        
        if self.similarity_metric == "cosine":
            faiss.normalize_L2(query_embeddings)
        
        distances, indices = self.index.search(query_embeddings, top_k)
        
        all_results = []
        for query_idx in range(len(query_embeddings)):
            results = []
            for idx, dist in zip(indices[query_idx], distances[query_idx]):
                if idx < 0:
                    continue
                
                result = {
                    'index': int(idx),
                    'distance': float(dist),
                    'score': 1.0 - dist if self.similarity_metric == "cosine" else 1.0 / (1.0 + dist)
                }
                
                if self.chunks and idx < len(self.chunks):
                    result['text'] = self.chunks[idx]
                
                if self.chunk_metadata and idx < len(self.chunk_metadata):
                    result['metadata'] = self.chunk_metadata[idx]
                
                results.append(result)
            
            all_results.append(results)
        
        return all_results
    
    def save_index(self, path: str):
        """Save FAISS index to disk"""
        if self.index is None:
            raise ValueError("No index to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
    
    def load_index(self, path: str):
        """Load FAISS index from disk"""
        self.index = faiss.read_index(str(path))
    
    def get_index_info(self) -> Dict:
        """Get index information"""
        return {
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'similarity_metric': self.similarity_metric,
            'total_vectors': self.index.ntotal if self.index else 0,
            'is_trained': self.index.is_trained if self.index else False
        }



