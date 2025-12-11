"""
RAG Production Pipeline
A production-ready RAG system with comprehensive evaluation
"""

__version__ = "1.0.0"

from .core import *

# Evaluation imports (optional - may have dependency conflicts)
try:
    from .evaluation import *
except ImportError as e:
    pass  # Evaluation modules loaded separately in scripts
