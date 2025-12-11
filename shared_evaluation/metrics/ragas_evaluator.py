"""
RAGAS Evaluator
Integrates RAGAS framework for RAG evaluation
"""

from typing import List, Dict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class RAGASEvaluator:
    """
    RAGAS (RAG Assessment) framework integration
    
    Evaluates:
    - Faithfulness: Factual consistency with context
    - Answer Relevancy: Relevance to question
    - Context Recall: How much context was needed
    - Context Precision: Precision of retrieved contexts
    """
    
    def __init__(self):
        """Initialize RAGAS evaluator"""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
                context_relevancy
            )
            self.evaluate = evaluate
            self.metrics = {
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
                'context_recall': context_recall,
                'context_precision': context_precision,
                'context_relevancy': context_relevancy
            }
            self.available = True
        except ImportError:
            print("⚠️  RAGAS not available. Install with: pip install ragas")
            self.available = False
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate batch using RAGAS metrics
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists (retrieved docs)
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Dictionary of RAGAS metrics
        """
        if not self.available:
            return {
                'ragas_faithfulness': 0.0,
                'ragas_answer_relevancy': 0.0,
                'ragas_context_recall': 0.0,
                'ragas_context_precision': 0.0,
                'ragas_context_relevancy': 0.0
            }
        
        try:
            from datasets import Dataset
            
            # Prepare data in RAGAS format
            data = {
                'question': questions,
                'answer': answers,
                'contexts': contexts,
            }
            
            if ground_truths:
                data['ground_truths'] = [[gt] for gt in ground_truths]
            
            dataset = Dataset.from_dict(data)
            
            # Select metrics based on available data
            metrics_to_use = [
                self.metrics['faithfulness'],
                self.metrics['answer_relevancy'],
                self.metrics['context_relevancy']
            ]
            
            if ground_truths:
                metrics_to_use.extend([
                    self.metrics['context_recall'],
                    self.metrics['context_precision']
                ])
            
            # Evaluate
            result = self.evaluate(dataset, metrics=metrics_to_use)
            
            # Format results
            formatted_results = {}
            for key, value in result.items():
                formatted_results[f'ragas_{key}'] = float(value)
            
            return formatted_results
            
        except Exception as e:
            print(f"⚠️  RAGAS evaluation failed: {e}")
            return {
                'ragas_faithfulness': 0.0,
                'ragas_answer_relevancy': 0.0,
                'ragas_context_recall': 0.0,
                'ragas_context_precision': 0.0,
                'ragas_context_relevancy': 0.0
            }
    
    def simple_evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> Dict[str, float]:
        """
        Evaluate single query
        
        Args:
            question: Question text
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary of RAGAS metrics
        """
        questions = [question]
        answers = [answer]
        contexts_list = [contexts]
        ground_truths = [ground_truth] if ground_truth else None
        
        return self.evaluate_batch(questions, answers, contexts_list, ground_truths)

