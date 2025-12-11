"""
Shared Evaluation Framework
Reusable across all 4 RAG projects:
1. Basic RAG
2. RAG with LLM Judge
3. Agentic RAG
4. Multi-modal Agent System

This module provides:
- Retrieval metrics (P@K, R@K, MRR, NDCG, MAP)
- Generation metrics (BLEU, ROUGE, F1, EM)
- RAGAS evaluation (faithfulness, relevancy)
- DeepEval evaluation (hallucination, bias)
- MLflow experiment tracking
- Visualization tools
- Export utilities
- Golden dataset management
"""

# Base evaluator (abstract class)
from .base_evaluator import BaseEvaluator

# Core evaluators
try:
    from .composite_evaluator import CompositeEvaluator
except ImportError as e:
    print(f"Note: CompositeEvaluator not available ({e})")
    CompositeEvaluator = None

# Metrics
try:
    from .metrics.retrieval_metrics import RetrievalMetrics
    from .metrics.generation_metrics import GenerationMetrics
except ImportError as e:
    print(f"Note: Metrics not fully available ({e})")
    RetrievalMetrics = None
    GenerationMetrics = None

# RAGAS and DeepEval (optional - require additional packages)
try:
    from .metrics.ragas_evaluator import RAGASEvaluator
except ImportError:
    RAGASEvaluator = None

try:
    from .metrics.deepeval_evaluator import DeepEvalEvaluator
except ImportError:
    DeepEvalEvaluator = None

# MLflow tracking (optional - may have protobuf conflicts)
try:
    from .mlflow_tracker import MLflowExperimentTracker
except ImportError as e:
    MLflowExperimentTracker = None

# Visualization (optional)
try:
    from .visualizer import RAGVisualizer
except ImportError:
    RAGVisualizer = None

# Export utilities
try:
    from .export_utils import ResultsExporter
except ImportError:
    ResultsExporter = None

# Golden dataset
try:
    from .golden_dataset import GoldenDatasetManager
except ImportError:
    GoldenDatasetManager = None

__version__ = "1.0.0"

__all__ = [
    # Base
    "BaseEvaluator",
    
    # Core
    "CompositeEvaluator",
    
    # Metrics
    "RetrievalMetrics",
    "GenerationMetrics",
    "RAGASEvaluator",
    "DeepEvalEvaluator",
    
    # Tracking & Export
    "MLflowExperimentTracker",
    "RAGVisualizer",
    "ResultsExporter",
    
    # Dataset
    "GoldenDatasetManager",
]
