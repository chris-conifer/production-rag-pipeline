"""
Generation Metrics
Quality metrics for generated text
"""

from typing import List, Dict
import numpy as np
from collections import Counter


class GenerationMetrics:
    """
    Text generation evaluation metrics
    
    Includes:
    - BLEU score
    - ROUGE scores (1, 2, L)
    - Exact Match
    - F1 Score (token-level)
    """
    
    @staticmethod
    def exact_match(prediction: str, reference: str) -> float:
        """
        Exact Match: 1 if prediction exactly matches reference, 0 otherwise
        
        Args:
            prediction: Generated text
            reference: Ground truth text
            
        Returns:
            Exact match score (0 or 1)
        """
        return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0
    
    @staticmethod
    def f1_score(prediction: str, reference: str) -> float:
        """
        Token-level F1 score
        
        Args:
            prediction: Generated text
            reference: Ground truth text
            
        Returns:
            F1 score [0, 1]
        """
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def bleu_score(prediction: str, reference: str, n: int = 1) -> float:
        """
        BLEU-N score (simplified)
        
        Args:
            prediction: Generated text
            reference: Ground truth text
            n: N-gram size
            
        Returns:
            BLEU score [0, 1]
        """
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if len(pred_tokens) < n or len(ref_tokens) < n:
            return 0.0
        
        # Create n-grams
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
        
        # Count matches
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        matches = sum((pred_counter & ref_counter).values())
        total = len(pred_ngrams)
        
        if total == 0:
            return 0.0
        
        return matches / total
    
    @staticmethod
    def rouge_n(prediction: str, reference: str, n: int = 1) -> Dict[str, float]:
        """
        ROUGE-N score
        
        Args:
            prediction: Generated text
            reference: Ground truth text
            n: N-gram size
            
        Returns:
            Dictionary with precision, recall, f1
        """
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if len(pred_tokens) < n or len(ref_tokens) < n:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Create n-grams
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
        
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        overlap = sum((pred_counter & ref_counter).values())
        
        precision = overlap / len(pred_ngrams) if len(pred_ngrams) > 0 else 0.0
        recall = overlap / len(ref_ngrams) if len(ref_ngrams) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    @staticmethod
    def rouge_l(prediction: str, reference: str) -> Dict[str, float]:
        """
        ROUGE-L: Longest Common Subsequence based score
        
        Args:
            prediction: Generated text
            reference: Ground truth text
            
        Returns:
            Dictionary with precision, recall, f1
        """
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        # Compute LCS length
        lcs_length = GenerationMetrics._lcs_length(pred_tokens, ref_tokens)
        
        precision = lcs_length / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def evaluate_batch(
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate batch of predictions
        
        Args:
            predictions: List of generated texts
            references: List of ground truth texts
            
        Returns:
            Dictionary of average metrics
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have same length")
        
        exact_matches = []
        f1_scores = []
        bleu_1_scores = []
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for pred, ref in zip(predictions, references):
            exact_matches.append(GenerationMetrics.exact_match(pred, ref))
            f1_scores.append(GenerationMetrics.f1_score(pred, ref))
            bleu_1_scores.append(GenerationMetrics.bleu_score(pred, ref, n=1))
            
            rouge_1 = GenerationMetrics.rouge_n(pred, ref, n=1)
            rouge_2 = GenerationMetrics.rouge_n(pred, ref, n=2)
            rouge_l = GenerationMetrics.rouge_l(pred, ref)
            
            rouge_1_scores.append(rouge_1['f1'])
            rouge_2_scores.append(rouge_2['f1'])
            rouge_l_scores.append(rouge_l['f1'])
        
        return {
            'exact_match': np.mean(exact_matches),
            'f1': np.mean(f1_scores),
            'bleu_1': np.mean(bleu_1_scores),
            'rouge_1': np.mean(rouge_1_scores),
            'rouge_2': np.mean(rouge_2_scores),
            'rouge_l': np.mean(rouge_l_scores)
        }

