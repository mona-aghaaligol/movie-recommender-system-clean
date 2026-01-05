"""
Backwards-compatible facade for SVD-related symbols.

Canonical implementation lives in:
    src.recommender.models.svd_model

This module exists only to preserve older import paths.
Do not add business logic here.
"""

from __future__ import annotations

from src.recommender.models.svd_model import MatrixFactorization, RatingsDataset

__all__ = ["RatingsDataset", "MatrixFactorization"]

