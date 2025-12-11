"""
Composite Evaluator
Orchestrates all evaluation components (retrieval, generation, RAGAS, DeepEval)
SHARED across all 4 projects
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .metrics.retrieval_metrics import RetrievalMetrics
from .metrics.generation_metrics import GenerationMetrics
from .metrics.ragas_evaluator import RAGASEvaluator
from .metrics.deepeval_evaluator import DeepEvalEvaluator


class CompositeEvaluator:
    """
    Orchestrates all evaluation metrics for RAG pipelines
    
    Combines:
    - Retrieval metrics (P@K, R@K, MRR, NDCG, MAP)
    - Generation metrics (BLEU, ROUGE, F1, EM)
    - RAGAS metrics (faithfulness, relevancy, etc.)
    - DeepEval metrics (hallucination, bias, etc.)
    
    SHARED across all 4 projects
    """
    
    def __init__(
        self,
        enable_ragas: bool = True,
        enable_deepeval: bool = True,
        ragas_metrics: List[str] = None,
        deepeval_metrics: List[str] = None
    ):
        """
        Initialize composite evaluator
        
        Args:
            enable_ragas: Whether to use RAGAS evaluation
            enable_deepeval: Whether to use DeepEval evaluation
            ragas_metrics: List of RAGAS metrics to use
            deepeval_metrics: List of DeepEval metrics to use
        """
        print("\n🔧 Initializing Composite Evaluator")
        
        # Core metrics (always enabled)
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        print("   ✓ Retrieval & Generation metrics loaded")
        
        # Optional frameworks
        self.ragas_evaluator = None
        self.deepeval_evaluator = None
        
        if enable_ragas:
            try:
                self.ragas_evaluator = RAGASEvaluator(
                    metrics=ragas_metrics or ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
                )
                print("   ✓ RAGAS evaluator loaded")
            except Exception as e:
                print(f"   ⚠️ RAGAS evaluator failed to load: {e}")
        
        if enable_deepeval:
            try:
                self.deepeval_evaluator = DeepEvalEvaluator(
                    metrics=deepeval_metrics or ['faithfulness', 'answer_relevancy', 'hallucination']
                )
                print("   ✓ DeepEval evaluator loaded")
            except Exception as e:
                print(f"   ⚠️ DeepEval evaluator failed to load: {e}")
        
        print("✓ Composite Evaluator ready\n")
    
    def evaluate_full_pipeline(
        self,
        queries: List[str],
        generated_answers: List[str],
        retrieved_contexts: List[List[str]],
        ground_truth_answers: List[List[str]],
        ground_truth_contexts: List[List[str]],
        retrieved_results_with_metadata: List[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate complete RAG pipeline with all metrics
        
        Args:
            queries: List of query strings
            generated_answers: List of generated answer strings
            retrieved_contexts: List of lists of retrieved context strings
            ground_truth_answers: List of lists of ground truth answer strings
            ground_truth_contexts: List of lists of ground truth context strings
            retrieved_results_with_metadata: Optional list of retrieval results with metadata
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE RAG EVALUATION")
        print("=" * 80)
        
        all_metrics = {}
        
        # 1. Retrieval Metrics
        print("\n📊 Evaluating Retrieval...")
        if retrieved_results_with_metadata:
            # Prepare ground truth data for retrieval evaluation
            ground_truth_data = [
                {
                    'question': queries[i],
                    'relevant_contexts': ground_truth_contexts[i],
                    'answers': ground_truth_answers[i]
                }
                for i in range(len(queries))
            ]
            
            retrieval_metrics = self.retrieval_metrics.calculate_retrieval_metrics(
                retrieved_results=retrieved_results_with_metadata,
                ground_truth_data=ground_truth_data,
                top_k_values=[1, 3, 5, 10]
            )
            all_metrics['retrieval'] = retrieval_metrics
            print(f"   ✓ Computed {len(retrieval_metrics)} retrieval metrics")
        else:
            print("   ⚠️ Skipped (no retrieved_results_with_metadata provided)")
        
        # 2. Generation Metrics
        print("\n📊 Evaluating Generation...")
        generation_metrics = self.generation_metrics.calculate_generation_metrics(
            predictions=generated_answers,
            references=ground_truth_answers
        )
        all_metrics['generation'] = generation_metrics
        print(f"   ✓ Computed {len(generation_metrics)} generation metrics")
        
        # 3. RAGAS Metrics
        if self.ragas_evaluator:
            print("\n📊 Evaluating with RAGAS...")
            try:
                ragas_metrics = self.ragas_evaluator.evaluate_ragas(
                    questions=queries,
                    answers=generated_answers,
                    contexts=retrieved_contexts,
                    ground_truths=ground_truth_answers
                )
                all_metrics['ragas'] = ragas_metrics
                print(f"   ✓ Computed RAGAS metrics")
            except Exception as e:
                print(f"   ⚠️ RAGAS evaluation failed: {e}")
                all_metrics['ragas'] = {'status': 'failed', 'error': str(e)}
        
        # 4. DeepEval Metrics
        if self.deepeval_evaluator:
            print("\n📊 Evaluating with DeepEval...")
            try:
                deepeval_metrics = self.deepeval_evaluator.evaluate_deepeval(
                    questions=queries,
                    answers=generated_answers,
                    contexts=retrieved_contexts,
                    ground_truths=ground_truth_answers
                )
                all_metrics['deepeval'] = deepeval_metrics
                print(f"   ✓ Computed DeepEval metrics")
            except Exception as e:
                print(f"   ⚠️ DeepEval evaluation failed: {e}")
                all_metrics['deepeval'] = {'status': 'failed', 'error': str(e)}
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80 + "\n")
        
        return all_metrics
    
    def compute_composite_accuracy(self, all_metrics: Dict[str, Any]) -> float:
        """
        Compute a single composite accuracy score from all metrics
        
        Args:
            all_metrics: Dictionary with all evaluation metrics
            
        Returns:
            Composite accuracy score (0-1)
        """
        scores = []
        weights = []
        
        # Retrieval metrics (weight: 0.3)
        if 'retrieval' in all_metrics and all_metrics['retrieval']:
            retrieval_score = np.mean([
                all_metrics['retrieval'].get('precision_at_5', 0),
                all_metrics['retrieval'].get('recall_at_5', 0),
                all_metrics['retrieval'].get('ndcg_at_5', 0)
            ])
            scores.append(retrieval_score)
            weights.append(0.3)
        
        # Generation metrics (weight: 0.4)
        if 'generation' in all_metrics and all_metrics['generation']:
            generation_score = np.mean([
                all_metrics['generation'].get('f1_score', 0),
                all_metrics['generation'].get('rouge1_f1', 0),
                all_metrics['generation'].get('exact_match', 0)
            ])
            scores.append(generation_score)
            weights.append(0.4)
        
        # RAGAS metrics (weight: 0.2)
        if 'ragas' in all_metrics and all_metrics['ragas'].get('ragas_status') == 'success':
            ragas_keys = [k for k in all_metrics['ragas'].keys() if k.startswith('ragas_') and k != 'ragas_status']
            if ragas_keys:
                ragas_score = np.mean([all_metrics['ragas'][k] for k in ragas_keys])
                scores.append(ragas_score)
                weights.append(0.2)
        
        # DeepEval metrics (weight: 0.1)
        if 'deepeval' in all_metrics and all_metrics['deepeval'].get('deepeval_status') == 'success':
            deepeval_keys = [k for k in all_metrics['deepeval'].keys() if k.startswith('deepeval_') and k != 'deepeval_status']
            if deepeval_keys:
                deepeval_score = np.mean([all_metrics['deepeval'][k] for k in deepeval_keys])
                scores.append(deepeval_score)
                weights.append(0.1)
        
        # Normalize weights
        if not scores:
            return 0.0
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        composite_accuracy = sum(s * w for s, w in zip(scores, normalized_weights))
        return composite_accuracy
    
    def format_metrics_summary(self, all_metrics: Dict[str, Any]) -> str:
        """
        Format metrics into a human-readable summary
        
        Args:
            all_metrics: Dictionary with all evaluation metrics
            
        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("EVALUATION METRICS SUMMARY")
        lines.append("=" * 80 + "\n")
        
        # Retrieval
        if 'retrieval' in all_metrics and all_metrics['retrieval']:
            lines.append("📊 Retrieval Metrics:")
            for k, v in all_metrics['retrieval'].items():
                lines.append(f"   {k}: {v:.4f}")
            lines.append("")
        
        # Generation
        if 'generation' in all_metrics and all_metrics['generation']:
            lines.append("📝 Generation Metrics:")
            for k, v in all_metrics['generation'].items():
                lines.append(f"   {k}: {v:.4f}")
            lines.append("")
        
        # RAGAS
        if 'ragas' in all_metrics:
            lines.append("🔍 RAGAS Metrics:")
            for k, v in all_metrics['ragas'].items():
                if isinstance(v, (int, float)):
                    lines.append(f"   {k}: {v:.4f}")
                else:
                    lines.append(f"   {k}: {v}")
            lines.append("")
        
        # DeepEval
        if 'deepeval' in all_metrics:
            lines.append("🧪 DeepEval Metrics:")
            for k, v in all_metrics['deepeval'].items():
                if isinstance(v, (int, float)):
                    lines.append(f"   {k}: {v:.4f}")
                else:
                    lines.append(f"   {k}: {v}")
            lines.append("")
        
        # Composite
        composite_accuracy = self.compute_composite_accuracy(all_metrics)
        lines.append(f"🎯 Composite Accuracy Score: {composite_accuracy:.4f}")
        lines.append("=" * 80 + "\n")
        
        return "\n".join(lines)



