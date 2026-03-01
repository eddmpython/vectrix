"""
Flat Prediction Defense System

4-stage defense:
1. FlatRiskDiagnostic - Pre-risk diagnosis
2. AdaptiveModelSelector - Adaptive model selection
3. FlatPredictionDetector - Post-forecast detection
4. FlatPredictionCorrector - Intelligent correction
"""

from .corrector import FlatPredictionCorrector
from .detector import FlatPredictionDetector
from .diagnostic import FlatRiskDiagnostic

__all__ = [
    "FlatRiskDiagnostic",
    "FlatPredictionDetector",
    "FlatPredictionCorrector"
]
