import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from .baselines import popularity_baseline, random_baseline
from .protocol import precision_at_k, recall_at_k


_ALLOWED_USER_COLUMNS = ("user_id", "userId")
_ALLOWED_MOVIE_COLUMNS = ("movie_id", "movieId")


def _normalize_ratings_schema(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Accept external schemas and normalize to internal evaluation schema:
      - user_id
      - movie_id
    """
    cols = set(ratings.columns)

    user_col = next((c for c in _ALLOWED_USER_COLUMNS if c in cols), None)
    movie_col = next((c for c in _ALLOWED_MOVIE_COLUMNS if c in cols), None)

    if user_col is None or movie_col is None:
        raise ValueError(
            "ratings must contain user and movie columns. "
            f"Expected user in {_ALLOWED_USER_COLUMNS} and movie in {_ALLOWED_MOVIE_COLUMNS}. "
            f"Found columns: {sorted(ratings.columns)}"
        )

    out = ratings.copy()
    if user_col != "user_id":
        out = out.rename(columns={user_col: "user_id"})
    if movie_col != "movie_id":
        out = out.rename(columns={movie_col: "movie_id"})

    # enforce int ids (CI-safe)
    out["user_id"] = out["user_id"].astype(int)
    out["movie_id"] = out["movie_id"].astype(int)
    return out


def _validate_ratings_schema(ratings: pd.DataFrame) -> None:
    required = {"user_id", "movie_id"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(
            f"ratings is missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(ratings.columns)}"
        )


def _build_per_user_holdout(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-out per user:
    - test: last interaction row per user (by row order)
    - train: all other rows
    Users with <2 interactions are excluded from evaluation (both train/test for that user).
    """
    counts = ratings.groupby("user_id").size()
    eligible_users = counts[counts >= 2].index

    eligible = ratings[ratings["user_id"].isin(eligible_users)].copy()

    last_idx = eligible.groupby("user_id").tail(1).index
    test = eligible.loc[last_idx].copy()
    train = eligible.drop(index=last_idx).copy()

    return train, test


def evaluate_baselines(
    ratings: pd.DataFrame,
    top_k: int = 10,
    seed: int = 42,
) -> Dict:
    ratings = _normalize_ratings_schema(ratings)
    _validate_ratings_schema(ratings)

    train, test = _build_per_user_holdout(ratings)

    # Single source of truth: use the baseline implementation from baselines.py
    unique_items = train["movie_id"].unique().tolist()
    candidate_item_ids: List[int] = popularity_baseline(
        train_ratings=train,
        top_k=len(unique_items),
    )

    results = {
        "random": {"precision@k": 0.0, "recall@k": 0.0},
        "popularity": {"precision@k": 0.0, "recall@k": 0.0},
        "meta": {"users_evaluated": 0, "k": top_k},
    }

    precision_random: List[float] = []
    recall_random: List[float] = []
    precision_pop: List[float] = []
    recall_pop: List[float] = []

    train_by_user = train.groupby("user_id")
    test_by_user = test.groupby("user_id")

    for user_id, test_df in test_by_user:
        test_item = int(test_df.iloc[0]["movie_id"])
        relevant: Set[int] = {test_item}

        user_train = train_by_user.get_group(user_id)
        seen: Set[int] = set(user_train["movie_id"].astype(int).tolist())

        # candidate items excluding what user has already seen in TRAIN
        available = [mid for mid in candidate_item_ids if mid not in seen]

        # Popularity baseline: take top-K from global ranking excluding seen
        pop_rec = available[:top_k]

        # Random baseline: sample from the same available set (deterministic per user)
        if len(available) >= top_k:
            rand_rec = random_baseline(available, top_k, seed=seed + int(user_id))
        else:
            rand_rec = available[:]

        precision_pop.append(precision_at_k(pop_rec, relevant, top_k))
        recall_pop.append(recall_at_k(pop_rec, relevant, top_k))

        precision_random.append(precision_at_k(rand_rec, relevant, top_k))
        recall_random.append(recall_at_k(rand_rec, relevant, top_k))

    if precision_random:
        results["random"]["precision@k"] = sum(precision_random) / len(precision_random)
        results["random"]["recall@k"] = sum(recall_random) / len(recall_random)
        results["popularity"]["precision@k"] = sum(precision_pop) / len(precision_pop)
        results["popularity"]["recall@k"] = sum(recall_pop) / len(recall_pop)
        results["meta"]["users_evaluated"] = len(precision_random)

    return results


def run_and_save(
    ratings: pd.DataFrame,
    output_path: str = "metrics.json",
    top_k: int = 10,
    seed: int = 42,
) -> Dict:
    metrics = evaluate_baselines(
        ratings=ratings,
        top_k=top_k,
        seed=seed,
    )

    path = Path(output_path)
    path.write_text(json.dumps(metrics, indent=2))
    return metrics
