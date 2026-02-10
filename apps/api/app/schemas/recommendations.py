"""
Pydantic schemas for recommendations endpoints.

- RecommendationOut: response item model for a single recommended movie.
- RecommendationQueryParams: query parameters for GET /v1/recommendations/{user_id}
  with validation (min/max) so Swagger/OpenAPI shows the allowed ranges.

Notes:
- The `score` field represents a *relative recommendation score*.
- If upstream model scores are degenerate (e.g., all equal),
  the service layer may derive this score from rank to keep results meaningful.
"""

from typing import Optional

from pydantic import BaseModel, Field


class RecommendationOut(BaseModel):
    """
    API response model for a single recommended movie.

    Attributes
    ----------
    movie_id:
        Internal movie identifier.
    title:
        Human-readable movie title.
    score:
        Relative recommendation score (higher is better).

        This is NOT guaranteed to be a predicted rating.
        If upstream model scores are not informative (e.g., all equal),
        this value may be derived from the recommendation rank to preserve
        meaningful ordering for clients and UIs.
    """

    movie_id: int = Field(
        ...,
        description="Internal movie identifier.",
        examples=[3404],
    )

    title: str = Field(
        ...,
        description="Movie title.",
        examples=["Titanic (1953)"],
    )

    score: float = Field(
        ...,
        description=(
            "Relative recommendation score (higher is better). "
            "If upstream model scores are degenerate, this value may be "
            "derived from rank to remain meaningful."
        ),
        examples=[1.0, 0.9, 0.8],
    )


class RecommendationQueryParams(BaseModel):
    """
    Query params for recommendations endpoint.

    Notes:
    - limit: how many results to return
    - neighbors_k: number of neighbors used by KNN-like logic (if applicable)
    - min_similarity: similarity threshold in [0.0, 1.0]
    - min_raters: minimum number of raters (integer count, not a rating score)
    """

    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="How many results to return (1 to 100).",
        examples=[10],
    )

    neighbors_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Number of neighbors to consider (1 to 200).",
        examples=[50],
    )

    min_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0.0 to 1.0).",
        examples=[0.2],
    )

    min_raters: Optional[int] = Field(
        default=None,
        ge=1,
        le=10000,
        description="Minimum number of raters required (1 to 10000).",
        examples=[50],
    )
