"""
Base Evaluator - Abstract class for all evaluation implementations
All 4 projects should inherit from this for consistency
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators
    
    All projects should implement this interface for consistent evaluation
    """
    
    @abstractmethod
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system performance
        
        Args:
            questions: List of input questions
            answers: List of generated answers  
            contexts: List of retrieved context lists
            ground_truths: Optional ground truth answers
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Return list of metric names this evaluator computes"""
        pass

