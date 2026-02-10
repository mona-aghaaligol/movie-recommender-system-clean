from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.recommender.config import load_app_config
from src.recommender.data.load_data.load_data_from_mongo import load_movies_and_ratings
from src.recommender.logging_utils import configure_logger
from src.recommender.service.recommender_service import RecommenderService

logger = configure_logger(__name__)


def _repo_root() -> Path:
    # .../src/recommender/bootstrap.py -> parents[2] = repo root
    return Path(__file__).resolve().parents[2]


def _load_similarity_csv(path: Path) -> pd.DataFrame:
    full_path = path if path.is_absolute() else (_repo_root() / path)

    if not full_path.exists():
        raise FileNotFoundError(f"Similarity matrix file not found: {full_path}")

    df = pd.read_csv(full_path, index_col=0)

    # Defensive: normalize ids to ints when possible.
    try:
        df.index = df.index.astype(int)
    except Exception:
        pass

    try:
        df.columns = df.columns.astype(int)
    except Exception:
        pass

    return df


def bootstrap_service() -> RecommenderService:
    """
    Build and return a fully initialized RecommenderService.

    This function performs I/O (Mongo reads, CSV loads) and is intended to be
    called during application startup to avoid cold-start latency on requests.
    """
    logger.info(
        "Bootstrapping RecommenderService",
        extra={"event": "bootstrap.start"},
    )

    app_config = load_app_config()

    movies_df, ratings_df = load_movies_and_ratings(
        config=app_config.mongo,
        movie_required_columns=("movieId", "title", "genres"),
        rating_required_columns=("userId", "movieId", "rating"),
    )

    # Type alignment
    movies_df["movieId"] = movies_df["movieId"].astype(int)
    ratings_df["userId"] = ratings_df["userId"].astype(int)
    ratings_df["movieId"] = ratings_df["movieId"].astype(int)

    user_user_sim = _load_similarity_csv(app_config.similarity.user_user_output_csv_path)
    item_item_sim = _load_similarity_csv(app_config.similarity.item_item_output_csv_path)

    service = RecommenderService(
        ratings_df=ratings_df,
        movies_df=movies_df,
        user_user_sim=user_user_sim,
        item_item_sim=item_item_sim,
        default_params=None,
    )

    logger.info(
        "RecommenderService initialized successfully",
        extra={"event": "bootstrap.ready"},
    )
    return service


@lru_cache(maxsize=1)
def get_recommender_service() -> RecommenderService:
    """
    Cached service getter for scripts/tests or as a safe fallback.

    The API should prefer startup warmup (app.state) to avoid request-time cold starts.
    """
    return bootstrap_service()
