from __future__ import annotations

import numpy as np
import pandas as pd


def build_user_user_similarity_from_ratings(
    ratings: pd.DataFrame,
    *,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
) -> pd.DataFrame:
    """
    Build a small user-user cosine similarity matrix from ratings.
    CI-safe: deterministic, in-memory, designed for small fixtures.

    Returns a square DataFrame:
      index = userIds
      columns = userIds
      values = cosine similarity in [0, 1]
    """
    # pivot to user x item
    pivot = ratings.pivot_table(
        index=user_col,
        columns=item_col,
        values=rating_col,
        aggfunc="mean",
        fill_value=0.0,
    ).astype(float)

    # cosine similarity: (A A^T) / (||A|| ||A||)
    A = pivot.to_numpy()
    norms = np.linalg.norm(A, axis=1)
    # avoid divide-by-zero
    norms[norms == 0.0] = 1.0
    sim = (A @ A.T) / (norms[:, None] * norms[None, :])

    users = pivot.index.astype(int)
    out = pd.DataFrame(sim, index=users, columns=users)
    return out

