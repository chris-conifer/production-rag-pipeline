"""
Project 1 Evaluation - Uses Shared Evaluation Framework
Imports from shared_evaluation for consistency across all 4 projects
"""

import sys
from pathlib import Path

# Add shared_evaluation to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from shared evaluation
from shared_evaluation import (
    BaseEvaluator,
    CompositeEvaluator,
    GoldenDatasetManager
)
from shared_evaluation.metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    RAGASEvaluator,
    DeepEvalEvaluator
)

__all__ = [
    "BaseEvaluator",
    "CompositeEvaluator",
    "GoldenDatasetManager",
    "RetrievalMetrics",
    "GenerationMetrics",
    "RAGASEvaluator",
    "DeepEvalEvaluator",
]
