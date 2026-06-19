"""
Self-convergence utilities for mixed-dimensional MPSA analysis.
"""

from .self_convergence_l2_p1 import (
    matrix_overlap_l2_components_p1,
    fracture_overlap_l2_components_p1,
)

__all__ = [
    "matrix_overlap_l2_components_p1",
    "fracture_overlap_l2_components_p1",
]
