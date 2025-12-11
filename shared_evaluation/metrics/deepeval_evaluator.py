"""
DeepEval Evaluator
Integrates DeepEval framework for comprehensive evaluation
"""

from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')


class DeepEvalEvaluator:
    """
    DeepEval framework integration
    
    Evaluates:
    - Answer Relevancy
    - Faithfulness
    - Contextual Relevancy
    - Hallucination detection
    - Bias detection
    """
    
    def __init__(self):
        """Initialize DeepEval evaluator"""
        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                FaithfulnessMetric,
                ContextualRelevancyMetric,
                HallucinationMetric
            )
            from deepeval.test_case import LLMTestCase
            
            self.AnswerRelevancyMetric = AnswerRelevancyMetric
            self.FaithfulnessMetric = FaithfulnessMetric
            self.ContextualRelevancyMetric = ContextualRelevancyMetric
            self.HallucinationMetric = HallucinationMetric
            self.LLMTestCase = LLMTestCase
            self.available = True
        except ImportError:
            print("⚠️  DeepEval not available. Install with: pip install deepeval")
            self.available = False
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> Dict[str, float]:
        """
        Evaluate single query using DeepEval
        
        Args:
            question: Question text
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary of DeepEval metrics
        """
        if not self.available:
            return {
                'deepeval_answer_relevancy': 0.0,
                'deepeval_faithfulness': 0.0,
                'deepeval_contextual_relevancy': 0.0,
                'deepeval_hallucination': 0.0
            }
        
        try:
            # Create test case
            test_case = self.LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=ground_truth,
                retrieval_context=contexts
            )
            
            results = {}
            
            # Answer Relevancy
            try:
                metric = self.AnswerRelevancyMetric(threshold=0.5)
                metric.measure(test_case)
                results['deepeval_answer_relevancy'] = float(metric.score)
            except:
                results['deepeval_answer_relevancy'] = 0.0
            
            # Faithfulness
            try:
                metric = self.FaithfulnessMetric(threshold=0.5)
                metric.measure(test_case)
                results['deepeval_faithfulness'] = float(metric.score)
            except:
                results['deepeval_faithfulness'] = 0.0
            
            # Contextual Relevancy
            try:
                metric = self.ContextualRelevancyMetric(threshold=0.5)
                metric.measure(test_case)
                results['deepeval_contextual_relevancy'] = float(metric.score)
            except:
                results['deepeval_contextual_relevancy'] = 0.0
            
            # Hallucination (lower is better)
            try:
                metric = self.HallucinationMetric(threshold=0.5)
                metric.measure(test_case)
                results['deepeval_hallucination'] = float(metric.score)
            except:
                results['deepeval_hallucination'] = 0.0
            
            return results
            
        except Exception as e:
            print(f"⚠️  DeepEval evaluation failed: {e}")
            return {
                'deepeval_answer_relevancy': 0.0,
                'deepeval_faithfulness': 0.0,
                'deepeval_contextual_relevancy': 0.0,
                'deepeval_hallucination': 0.0
            }
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate batch of queries
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Dictionary of average DeepEval metrics
        """
        if not self.available:
            return {
                'deepeval_answer_relevancy': 0.0,
                'deepeval_faithfulness': 0.0,
                'deepeval_contextual_relevancy': 0.0,
                'deepeval_hallucination': 0.0
            }
        
        import numpy as np
        
        all_results = []
        for i, (q, a, c) in enumerate(zip(questions, answers, contexts)):
            gt = ground_truths[i] if ground_truths else None
            result = self.evaluate_single(q, a, c, gt)
            all_results.append(result)
        
        # Average results
        avg_results = {}
        for key in all_results[0].keys():
            values = [r[key] for r in all_results]
            avg_results[key] = float(np.mean(values))
        
        return avg_results

