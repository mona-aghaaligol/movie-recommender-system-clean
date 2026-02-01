"""
Service layer for the movie recommender system.

This module defines the high-level service interface that will be consumed
by upper layers such as FastAPI, CLIs, or batch pipelines.

Important:
    - This module does NOT perform any I/O (no DB, no CSV, no config loading).
    - This module does NOT own ML/CF logic.
    - This module ONLY orchestrates domain/core functionality and exposes
      type-safe contracts for recommendation-related use cases.

Data and algorithm execution dependencies are injected from the outside
(e.g., in application bootstrap code or API wiring).
"""

from dataclasses import dataclass, replace
from typing import List, Optional

import pandas as pd

from ..domain.recommend_for_user import (
    recommend_for_user,
    RecommendParams,
)


# ---------------------------------------------------------------------
# Typed Return Models
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Recommendation:
    """
    Typed model representing a recommended movie for a user.

    Attributes
    ----------
    movie_id:
        Internal movie identifier (as used in the dataset/core).
    score:
        Relevance score for the recommendation. Higher means better.
    title:
        Optional movie title, if available from movies metadata.
    """
    movie_id: int
    score: float
    title: Optional[str] = None


@dataclass(frozen=True)
class SimilarMovie:
    """
    Typed model representing a movie similar to another reference movie.

    Attributes
    ----------
    movie_id:
        Internal movie identifier of the similar movie.
    similarity:
        Numeric similarity coefficient (e.g., cosine similarity).
        Range and interpretation depend on the similarity method.
    """
    movie_id: int
    similarity: float


# ---------------------------------------------------------------------
# Service Layer
# ---------------------------------------------------------------------


class RecommenderService:
    """
    High-level service for user-facing recommendation use cases.

    This service is intentionally thin. It exposes contracts for retrieving
    recommendations and similarity results, while delegating the actual
    collaborative filtering and ranking logic to the domain/core layer.

    Design principles:
        - No I/O in this layer.
        - No direct database or CSV access.
        - No model training or heavy computation.
        - Pure orchestration of already-prepared data and domain functions.
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        user_user_sim: pd.DataFrame,
        item_item_sim: pd.DataFrame,
        default_params: Optional[RecommendParams] = None,
    ) -> None:
        """
        Construct a RecommenderService instance.

        Parameters
        ----------
        ratings_df:
            Pre-loaded ratings data as a pandas DataFrame.
        movies_df:
            Pre-loaded movies metadata as a pandas DataFrame.
        user_user_sim:
            Pre-loaded user-user similarity matrix as a pandas DataFrame.
        item_item_sim:
            Pre-loaded item-item similarity matrix as a pandas DataFrame.
        default_params:
            Optional default recommendation hyperparameters. If None,
            the domain-level defaults will be used.
        """
        self._ratings_df = ratings_df
        self._movies_df = movies_df
        self._user_user_sim = user_user_sim
        self._item_item_sim = item_item_sim
        self._default_params = default_params

    # ------------------------------------------------------------------
    # Public API - Recommendations
    # ------------------------------------------------------------------

    def get_recommendations_for_user(
        self,
        user_id: int,
        limit: int = 10,
        min_score: Optional[float] = None,
        params: Optional[RecommendParams] = None,
    ) -> List[Recommendation]:
        """
        Retrieve recommendations for a given user.

        This method orchestrates the call to the domain/core layer
        (e.g., collaborative filtering + ranking) and maps the result
        into a type-safe list of Recommendation objects.

        Parameters
        ----------
        user_id:
            User identifier in the dataset/core.
        limit:
            Maximum number of recommendations to return.
        min_score:
            Optional score threshold. If provided, recommendations
            with scores below this value will be filtered out.
        params:
            Optional per-call recommendation parameters. If not provided,
            the service's default_params are used (if any), otherwise
            domain defaults are used.

        Returns
        -------
        List[Recommendation]
            List of typed Recommendation objects. Length is at most `limit`.
        """
        # Resolve effective params: call-level > default > domain defaults
        effective_params = params or self._default_params or RecommendParams()

        # Ensure top_k is aligned with the service-level limit
        if effective_params.top_k != limit:
            effective_params = replace(effective_params, top_k=limit)

        domain_df = recommend_for_user(
            user_id=user_id,
            movies_df=self._movies_df,
            ratings_df=self._ratings_df,
            user_user_sim=self._user_user_sim,
            item_item_sim=self._item_item_sim,
            params=effective_params,
        )

        # Optionally filter by minimum score, if the score column exists.
        if min_score is not None and "score" in domain_df.columns:
            domain_df = domain_df[domain_df["score"] >= min_score]

        # Map the domain DataFrame to a list of typed Recommendation objects.
        # Expected columns: movieId, score, support, reasons, optional title.
        has_title = "title" in domain_df.columns

        recommendations: List[Recommendation] = []
        for _, row in domain_df.iterrows():
            title_val: Optional[str] = None
            if has_title:
                val = row["title"]
                if isinstance(val, str):
                    title_val = val
            recommendations.append(
                Recommendation(
                    movie_id=int(row["movieId"]),
                    score=float(row["score"]),
                    title=title_val,
                )
            )

        # Enforce the `limit` at the service boundary as well.
        return recommendations[:limit]

    # ------------------------------------------------------------------
    # Public API - Similarity (placeholder, to be wired later)
    # ------------------------------------------------------------------

    def get_similar_movies(
        self,
        movie_id: int,
        limit: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[SimilarMovie]:
        """
        Retrieve movies similar to a given reference movie.

        Parameters
        ----------
        movie_id:
            Movie identifier to compute similarity against.
        limit:
            Maximum number of similar movies to return.
        min_similarity:
            Optional threshold to filter out weak similarities.

        Returns
        -------
        List[SimilarMovie]
            List of typed SimilarMovie objects. Length is at most `limit`.

        Notes
        -----
        This is a placeholder. In a later iteration, this method will invoke
        a similarity engine from the domain/core layer (e.g., cosine similarity
        over a precomputed similarity matrix).
        """
        raise NotImplementedError(
            "get_similar_movies is not implemented yet. "
            "It will be wired to domain/core similarity computations "
            "in a later iteration."
        )
