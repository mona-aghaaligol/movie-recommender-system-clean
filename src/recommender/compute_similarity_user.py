from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .similarity_engine import UserUserCosineSimilarityEngine


def compute_user_user_similarity(
    ratings_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Compute the user–user similarity matrix from the given ratings DataFrame.

    This is a pure domain function:
        - It does NOT talk to MongoDB.
        - It does NOT load configuration.
        - It does NOT write to disk.
        - It only uses the similarity engine to transform ratings -> similarity.

    Parameters
    ----------
    ratings_df:
        A DataFrame of user-item ratings. Typically with columns like:
            - userId
            - movieId
            - rating
            - (optionally) timestamp

    logger:
        Optional logger instance. If not provided, a default module-level logger
        will be used.

    Returns
    -------
    pd.DataFrame
        A user–user similarity matrix, where:
            - index: userId
            - columns: userId
            - values: similarity scores (e.g., cosine similarity).
    """
    effective_logger = logger or logging.getLogger("recommender.similarity.user_user")

    engine = UserUserCosineSimilarityEngine(logger=effective_logger)
    similarity_df = engine.run_full_pipeline(ratings_df)

    return similarity_df
