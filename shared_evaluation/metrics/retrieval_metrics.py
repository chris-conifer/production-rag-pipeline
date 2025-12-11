"""
Retrieval Metrics
Ranking and non-ranking metrics for retrieval evaluation
"""

from typing import List, Dict, Set
import numpy as np


class RetrievalMetrics:
    """
    Comprehensive retrieval evaluation metrics
    
    Includes:
    - Precision@K, Recall@K, F1@K
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (NDCG)
    - Mean Average Precision (MAP)
    - Hit Rate
    """
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Precision@K: Proportion of retrieved documents that are relevant
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff position
            
        Returns:
            Precision@K score [0, 1]
        """
        if k <= 0 or not retrieved:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len([doc for doc in retrieved_at_k if doc in relevant])
        
        return relevant_retrieved / k
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Recall@K: Proportion of relevant documents that were retrieved
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff position
            
        Returns:
            Recall@K score [0, 1]
        """
        if not relevant or k <= 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len([doc for doc in retrieved_at_k if doc in relevant])
        
        return relevant_retrieved / len(relevant)
    
    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        F1@K: Harmonic mean of Precision@K and Recall@K
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff position
            
        Returns:
            F1@K score [0, 1]
        """
        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mrr(retrieved: List[str], relevant: Set[str]) -> float:
        """
        Mean Reciprocal Rank: Inverse of rank of first relevant document
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            
        Returns:
            MRR score [0, 1]
        """
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / rank
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff position
            
        Returns:
            NDCG@K score [0, 1]
        """
        if k <= 0 or not relevant:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for rank, doc in enumerate(retrieved[:k], 1):
            if doc in relevant:
                dcg += 1.0 / np.log2(rank + 1)
        
        # Calculate IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
        """
        Average Precision: Mean of precision at each relevant document position
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            
        Returns:
            AP score [0, 1]
        """
        if not relevant:
            return 0.0
        
        score = 0.0
        num_relevant_found = 0
        
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant:
                num_relevant_found += 1
                precision_at_rank = num_relevant_found / rank
                score += precision_at_rank
        
        return score / len(relevant) if len(relevant) > 0 else 0.0
    
    @staticmethod
    def hit_rate_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Hit Rate@K: 1 if any relevant document in top-k, 0 otherwise
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff position
            
        Returns:
            Hit rate (0 or 1)
        """
        if k <= 0:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        return 1.0 if len(retrieved_at_k & relevant) > 0 else 0.0
    
    @staticmethod
    def evaluate_batch(
        retrieved_lists: List[List[str]],
        relevant_lists: List[Set[str]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate batch of queries and compute average metrics
        
        Args:
            retrieved_lists: List of retrieved document lists
            relevant_lists: List of relevant document sets
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary of average metrics
        """
        if len(retrieved_lists) != len(relevant_lists):
            raise ValueError("retrieved_lists and relevant_lists must have same length")
        
        metrics = {}
        n = len(retrieved_lists)
        
        # Compute metrics for each k
        for k in k_values:
            precision_scores = []
            recall_scores = []
            f1_scores = []
            ndcg_scores = []
            hit_scores = []
            
            for retrieved, relevant in zip(retrieved_lists, relevant_lists):
                precision_scores.append(RetrievalMetrics.precision_at_k(retrieved, relevant, k))
                recall_scores.append(RetrievalMetrics.recall_at_k(retrieved, relevant, k))
                f1_scores.append(RetrievalMetrics.f1_at_k(retrieved, relevant, k))
                ndcg_scores.append(RetrievalMetrics.ndcg_at_k(retrieved, relevant, k))
                hit_scores.append(RetrievalMetrics.hit_rate_at_k(retrieved, relevant, k))
            
            metrics[f'precision@{k}'] = np.mean(precision_scores)
            metrics[f'recall@{k}'] = np.mean(recall_scores)
            metrics[f'f1@{k}'] = np.mean(f1_scores)
            metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
            metrics[f'hit_rate@{k}'] = np.mean(hit_scores)
        
        # Compute MRR and MAP
        mrr_scores = [RetrievalMetrics.mrr(retrieved, relevant)
                     for retrieved, relevant in zip(retrieved_lists, relevant_lists)]
        map_scores = [RetrievalMetrics.average_precision(retrieved, relevant)
                     for retrieved, relevant in zip(retrieved_lists, relevant_lists)]
        
        metrics['mrr'] = np.mean(mrr_scores)
        metrics['map'] = np.mean(map_scores)
        
        return metrics

