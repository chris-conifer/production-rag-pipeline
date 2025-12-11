"""
Hybrid Retriever
Combines dense (vector) and sparse (BM25) retrieval for improved performance
"""

from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
import faiss


class HybridRetriever:
    """
    Hybrid retrieval combining:
    - Dense retrieval: Vector similarity (FAISS)
    - Sparse retrieval: BM25 keyword matching
    
    Fusion strategies:
    - weighted: Weighted combination of scores
    - rrf: Reciprocal Rank Fusion
    - linear: Linear combination
    """
    
    def __init__(
        self,
        embedding_dimension: int,
        index_type: str = "IndexFlatL2",
        fusion_strategy: str = "rrf",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever
        
        Args:
            embedding_dimension: Dimension of embeddings
            index_type: FAISS index type
            fusion_strategy: How to combine dense + sparse scores
            dense_weight: Weight for dense retrieval (0-1)
            sparse_weight: Weight for sparse retrieval (0-1)
        """
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.fusion_strategy = fusion_strategy
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Dense retrieval (FAISS)
        self.faiss_index = self._initialize_faiss_index()
        self.chunk_map: List[Dict[str, Any]] = []
        
        # Sparse retrieval (BM25)
        self.bm25 = None
        self.tokenized_corpus = []
        
        print(f"Initialized HybridRetriever:")
        print(f"  Dense: FAISS {index_type}")
        print(f"  Sparse: BM25")
        print(f"  Fusion: {fusion_strategy} (dense={dense_weight}, sparse={sparse_weight})")
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index for dense retrieval"""
        if self.index_type == "IndexFlatL2":
            return faiss.IndexFlatL2(self.embedding_dimension)
        elif self.index_type == "IndexFlatIP":
            return faiss.IndexFlatIP(self.embedding_dimension)
        elif self.index_type == "IndexIDMap2":
            base_index = faiss.IndexFlatL2(self.embedding_dimension)
            return faiss.IndexIDMap2(base_index)
        elif self.index_type == "IndexIVFFlat":
            nlist = 100
            quantizer = faiss.IndexFlatL2(self.embedding_dimension)
            return faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist, faiss.METRIC_L2)
        elif self.index_type == "IndexHNSWFlat":
            M = 32
            return faiss.IndexHNSWFlat(self.embedding_dimension, M, faiss.METRIC_L2)
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.index_type}")
    
    def build_index(self, embeddings: np.ndarray, chunk_map: List[Dict[str, Any]]):
        """
        Build both dense and sparse indexes
        
        Args:
            embeddings: Dense embeddings (N, D)
            chunk_map: Metadata for each chunk (must include 'content')
        """
        print(f"\n🔨 Building Hybrid Index...")
        
        # Build dense index (FAISS)
        if self.index_type == "IndexIVFFlat" and not self.faiss_index.is_trained:
            print("  Training IVF index...")
            self.faiss_index.train(embeddings)
        
        self.faiss_index.add(embeddings)
        self.chunk_map = chunk_map
        print(f"  ✓ Dense index: {self.faiss_index.ntotal} vectors")
        
        # Build sparse index (BM25)
        corpus = [chunk['content'] for chunk in chunk_map]
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"  ✓ Sparse index: {len(self.tokenized_corpus)} documents")
        print(f"✓ Hybrid index ready\n")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        query_text: str,
        top_k: int = 10,
        dense_only: bool = False,
        sparse_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining dense and sparse retrieval
        
        Args:
            query_embedding: Dense query embedding
            query_text: Raw query text for BM25
            top_k: Number of results to return
            dense_only: Only use dense retrieval
            sparse_only: Only use sparse retrieval
            
        Returns:
            List of results with hybrid scores
        """
        if dense_only:
            return self._dense_search(query_embedding, top_k)
        elif sparse_only:
            return self._sparse_search(query_text, top_k)
        else:
            return self._hybrid_search(query_embedding, query_text, top_k)
    
    def _dense_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Dense retrieval using FAISS"""
        if self.faiss_index.ntotal == 0:
            return []
        
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx == -1:
                continue
            
            results.append({
                "chunk_content": self.chunk_map[idx]['content'],
                "chunk_metadata": self.chunk_map[idx]['metadata'],
                "dense_score": float(distances[0][i]),
                "sparse_score": 0.0,
                "hybrid_score": float(distances[0][i]),
                "retrieval_method": "dense"
            })
        
        return results
    
    def _sparse_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Sparse retrieval using BM25"""
        if not self.bm25:
            return []
        
        tokenized_query = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk_content": self.chunk_map[idx]['content'],
                "chunk_metadata": self.chunk_map[idx]['metadata'],
                "dense_score": 0.0,
                "sparse_score": float(bm25_scores[idx]),
                "hybrid_score": float(bm25_scores[idx]),
                "retrieval_method": "sparse"
            })
        
        return results
    
    def _hybrid_search(
        self, 
        query_embedding: np.ndarray, 
        query_text: str, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining dense and sparse
        Uses fusion strategy to combine scores
        """
        # Get more candidates from each method
        k_candidates = top_k * 2
        
        # Dense retrieval
        dense_distances, dense_indices = self.faiss_index.search(query_embedding, k_candidates)
        
        # Sparse retrieval
        tokenized_query = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Create score maps
        dense_score_map = {}
        for i in range(len(dense_indices[0])):
            idx = dense_indices[0][i]
            if idx != -1:
                # Convert distance to similarity score (inverse for L2)
                dense_score_map[idx] = 1.0 / (1.0 + float(dense_distances[0][i]))
        
        sparse_score_map = {i: float(score) for i, score in enumerate(bm25_scores)}
        
        # Normalize scores
        dense_score_map = self._normalize_scores(dense_score_map)
        sparse_score_map = self._normalize_scores(sparse_score_map)
        
        # Fusion
        if self.fusion_strategy == "weighted":
            hybrid_scores = self._weighted_fusion(dense_score_map, sparse_score_map)
        elif self.fusion_strategy == "rrf":
            hybrid_scores = self._reciprocal_rank_fusion(dense_score_map, sparse_score_map)
        elif self.fusion_strategy == "linear":
            hybrid_scores = self._linear_fusion(dense_score_map, sparse_score_map)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        # Sort by hybrid score
        sorted_indices = sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True)
        
        # Build results
        results = []
        for idx in sorted_indices[:top_k]:
            results.append({
                "chunk_content": self.chunk_map[idx]['content'],
                "chunk_metadata": self.chunk_map[idx]['metadata'],
                "dense_score": dense_score_map.get(idx, 0.0),
                "sparse_score": sparse_score_map.get(idx, 0.0),
                "hybrid_score": hybrid_scores[idx],
                "retrieval_method": "hybrid"
            })
        
        return results
    
    def _normalize_scores(self, score_map: Dict[int, float]) -> Dict[int, float]:
        """Min-max normalization of scores"""
        if not score_map:
            return {}
        
        scores = list(score_map.values())
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score - min_score < 1e-9:
            return {k: 1.0 for k in score_map.keys()}
        
        return {
            k: (v - min_score) / (max_score - min_score)
            for k, v in score_map.items()
        }
    
    def _weighted_fusion(
        self, 
        dense_scores: Dict[int, float], 
        sparse_scores: Dict[int, float]
    ) -> Dict[int, float]:
        """Weighted combination of scores"""
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        hybrid_scores = {}
        for idx in all_indices:
            dense = dense_scores.get(idx, 0.0)
            sparse = sparse_scores.get(idx, 0.0)
            hybrid_scores[idx] = self.dense_weight * dense + self.sparse_weight * sparse
        
        return hybrid_scores
    
    def _reciprocal_rank_fusion(
        self, 
        dense_scores: Dict[int, float], 
        sparse_scores: Dict[int, float],
        k: int = 60
    ) -> Dict[int, float]:
        """
        Reciprocal Rank Fusion (RRF)
        RRF(d) = sum(1 / (k + rank_i(d)))
        """
        # Get rankings
        dense_ranking = {idx: rank for rank, idx in enumerate(
            sorted(dense_scores.keys(), key=lambda x: dense_scores[x], reverse=True)
        )}
        sparse_ranking = {idx: rank for rank, idx in enumerate(
            sorted(sparse_scores.keys(), key=lambda x: sparse_scores[x], reverse=True)
        )}
        
        all_indices = set(dense_ranking.keys()) | set(sparse_ranking.keys())
        
        rrf_scores = {}
        for idx in all_indices:
            dense_rank = dense_ranking.get(idx, 1000)  # Penalty for missing
            sparse_rank = sparse_ranking.get(idx, 1000)
            rrf_scores[idx] = (1.0 / (k + dense_rank)) + (1.0 / (k + sparse_rank))
        
        return rrf_scores
    
    def _linear_fusion(
        self, 
        dense_scores: Dict[int, float], 
        sparse_scores: Dict[int, float]
    ) -> Dict[int, float]:
        """Simple linear combination (same as weighted but fixed 0.5/0.5)"""
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        hybrid_scores = {}
        for idx in all_indices:
            dense = dense_scores.get(idx, 0.0)
            sparse = sparse_scores.get(idx, 0.0)
            hybrid_scores[idx] = 0.5 * dense + 0.5 * sparse
        
        return hybrid_scores
    
    def save_index(self, faiss_path: str, bm25_path: str):
        """Save both FAISS and BM25 indexes"""
        import pickle
        
        # Save FAISS
        faiss.write_index(self.faiss_index, faiss_path)
        
        # Save BM25 (pickle)
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'tokenized_corpus': self.tokenized_corpus
            }, f)
        
        print(f"✓ Saved hybrid index")
    
    def load_index(self, faiss_path: str, bm25_path: str):
        """Load both FAISS and BM25 indexes"""
        import pickle
        
        # Load FAISS
        self.faiss_index = faiss.read_index(faiss_path)
        
        # Load BM25
        with open(bm25_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.tokenized_corpus = data['tokenized_corpus']
        
        print(f"✓ Loaded hybrid index")



