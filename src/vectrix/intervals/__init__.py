"""
Advanced Prediction Intervals

- Conformal: distribution-free valid intervals
- Bootstrap: residual bootstrap intervals
"""

from .bootstrap import BootstrapInterval
from .conformal import ConformalInterval

__all__ = [
    "ConformalInterval",
    "BootstrapInterval",
]
