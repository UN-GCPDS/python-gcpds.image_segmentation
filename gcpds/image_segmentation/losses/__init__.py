"""
=================================
Losses for Semantic Segmentation
=================================
"""

from .dice import DiceCoefficient, SparseCategoricalDiceCoefficient
from .generalized_cross_entropy import GeneralizedCrossEntropy