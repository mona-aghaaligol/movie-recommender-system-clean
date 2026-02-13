from __future__ import annotations

from typing import List
import random

import pandas as pd


# BigTech-style: normalize to one internal schema for evaluation code.
# We keep it minimal and explicit to avoid hidden magic.
_ALLOWED_MOVIE_COLUMNS = ("movie_id", "movieId")
_ALLOWED_USER_COLUMNS = ("user_id", "userId")


def _get_col(df: pd.DataFrame, candidates: tuple[str, ...], logical_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Missing required column for '{logical_name}'. "
        f"Expected one of {candidates}, got {list(df.columns)}"
    )


def popularity_baseline(
    train_ratings: pd.DataFrame,
    top_k: int,
) -> List[int]:
    """
    Popularity baseline (Top-K):
    Recommend the most frequently interacted-with items in the train data.

    CI-safe:
      - deterministic (purely count-based)
      - schema-safe (accepts movie_id or movieId)
    """
    movie_col = _get_col(train_ratings, _ALLOWED_MOVIE_COLUMNS, "movie_id")

    popularity = train_ratings[movie_col].value_counts().sort_values(ascending=False)
    return popularity.head(top_k).index.astype(int).tolist()


def random_baseline(
    candidate_item_ids: List[int],
    top_k: int,
    seed: int = 42,
) -> List[int]:
    """
    Random baseline (Top-K):
    Recommend K random items from a candidate pool.

    CI-safe:
      - deterministic (uses local RNG, no global random.seed)
    """
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if top_k > len(candidate_item_ids):
        raise ValueError(
            f"top_k ({top_k}) cannot exceed candidate pool size ({len(candidate_item_ids)})"
        )

    rng = random.Random(seed)
    return rng.sample(candidate_item_ids, k=top_k)

