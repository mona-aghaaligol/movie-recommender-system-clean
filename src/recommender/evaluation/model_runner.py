from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import pandas as pd

from src.recommender.service.recommender_service import RecommenderService
from src.recommender.domain.recommend_for_user import RecommendParams
from src.recommender.evaluation.protocol import precision_at_k, recall_at_k
from src.recommender.evaluation.similarity_small import build_user_user_similarity_from_ratings


def _build_per_user_holdout(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Same logic as baseline runner, but schema for model uses userId/movieId
    counts = ratings.groupby("userId").size()
    eligible_users = counts[counts >= 2].index
    eligible = ratings[ratings["userId"].isin(eligible_users)].copy()

    last_idx = eligible.groupby("userId").tail(1).index
    test = eligible.loc[last_idx].copy()
    train = eligible.drop(index=last_idx).copy()
    return train, test


@dataclass(frozen=True)
class ModelEvalResult:
    precision_at_k: float
    recall_at_k: float
    users_evaluated: int
    k: int


def evaluate_model_on_holdout(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    top_k: int = 10,
    seed: int = 42,
) -> Dict:
    """
    CI-safe model evaluation:
    - Build small user-user similarity from TRAIN only (avoid leakage)
    - Use alpha=1.0 (user-user only) to avoid huge item-item matrix
    - Evaluate Precision@K / Recall@K on per-user leave-one-out holdout
    """
    # normalize types
    ratings = ratings.copy()
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)

    movies = movies.copy()
    if "movieId" in movies.columns:
        movies["movieId"] = movies["movieId"].astype(int)

    train, test = _build_per_user_holdout(ratings)

    # Build similarity from TRAIN only
    user_user_sim = build_user_user_similarity_from_ratings(train)

    # Minimal item-item for alpha=1.0 (not used)
    item_item_sim = pd.DataFrame()

    # Build service with TRAIN data only
    svc = RecommenderService(
        ratings_df=train,
        movies_df=movies,
        user_user_sim=user_user_sim,
        item_item_sim=item_item_sim,
        default_params=None,
    )

    train_by_user = train.groupby("userId")
    test_by_user = test.groupby("userId")

    precs: List[float] = []
    recs: List[float] = []

    for user_id, test_df in test_by_user:
        test_item = int(test_df.iloc[0]["movieId"])
        relevant: Set[int] = {test_item}

        user_train = train_by_user.get_group(user_id)
        seen: Set[int] = set(user_train["movieId"].astype(int).tolist())

        # ask the model for recommendations
        params = RecommendParams(top_k=top_k, alpha=1.0)
        recs_obj = svc.get_recommendations_for_user(
            user_id=int(user_id),
            limit=top_k,
            params=params,
        )

        rec_ids = [int(r.movie_id) for r in recs_obj if int(r.movie_id) not in seen]
        rec_ids = rec_ids[:top_k]

        precs.append(precision_at_k(rec_ids, relevant, top_k))
        recs.append(recall_at_k(rec_ids, relevant, top_k))

    if not precs:
        return {
            "model": {"precision@k": 0.0, "recall@k": 0.0},
            "meta": {"users_evaluated": 0, "k": top_k},
        }

    return {
        "model": {
            "precision@k": sum(precs) / len(precs),
            "recall@k": sum(recs) / len(recs),
        },
        "meta": {"users_evaluated": len(precs), "k": top_k},
    }

