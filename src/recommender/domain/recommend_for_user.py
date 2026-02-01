from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecommendParams:
    """
    Hyperparameters for collaborative-filtering based recommendations.

    This dataclass is part of the pure *Domain / Code Layer*:
    it only carries configuration values and has no side effects.
    """
    top_k: int = 10
    neighbors_k: int = 50              # how many similar users/items to consider
    min_similarity: float = 0.0        # ignore weak/negative sims if you want
    min_raters: int = 1                # require at least N contributing ratings
    alpha: float = 1.0                 # 1.0 => only user-user, 0.0 => only item-item

    # --- Explainability ---
    explain: bool = False
    max_reasons: int = 3


class RecommendationError(ValueError):
    """
    Raised when recommendation inputs are invalid or the user
    cannot be recommended.

    This exception class belongs to the pure *Domain / Code Layer*.
    """


def _validate_inputs(
    user_id: int,
    ratings_df: pd.DataFrame,
    user_user_sim: pd.DataFrame,
    item_item_sim: Optional[pd.DataFrame],
    params: RecommendParams,
) -> None:
    """
    Validate inputs for the recommendation algorithm.

    Pure validation logic: raises RecommendationError on invalid inputs,
    but performs no I/O or logging.
    """
    required_cols = {"userId", "movieId", "rating"}
    missing = required_cols - set(ratings_df.columns)
    if missing:
        raise RecommendationError(f"ratings_df missing required columns: {sorted(missing)}")

    if params.top_k <= 0:
        raise RecommendationError("top_k must be > 0")
    if params.neighbors_k <= 0:
        raise RecommendationError("neighbors_k must be > 0")
    if not (0.0 <= params.alpha <= 1.0):
        raise RecommendationError("alpha must be in [0, 1]")

    if user_id not in set(ratings_df["userId"].unique()):
        raise RecommendationError(f"user_id={user_id} not found in ratings_df")

    # require user present in user-user similarity
    if user_id not in user_user_sim.index:
        raise RecommendationError(f"user_id={user_id} not found in user_user_sim matrix")

    # if hybrid or item-item only, item-item matrix must exist
    if params.alpha < 1.0 and item_item_sim is None:
        raise RecommendationError("item_item_sim is required when alpha < 1.0")


def _user_user_scores(
    user_id: int,
    ratings_df: pd.DataFrame,
    user_user_sim: pd.DataFrame,
    params: RecommendParams,
) -> pd.DataFrame:
    """
    Predict scores for unseen movies using a user-user CF scheme.

    score(m) = sum(sim(u,v) * r(v,m)) / sum(|sim(u,v)|)

    Returns a DataFrame with columns:
      - movieId
      - score
      - support
      - reasons: list[dict] when params.explain=True, otherwise None

    Pure in-memory computation: no I/O, no logging.
    """
    # Movies already rated by the target user
    user_rated = ratings_df.loc[ratings_df["userId"] == user_id, ["movieId"]]
    seen_movie_ids = set(user_rated["movieId"].tolist())

    # Similarities to other users (exclude self)
    sims = user_user_sim.loc[user_id].drop(index=user_id, errors="ignore")
    sims = sims.sort_values(ascending=False)

    # keep top neighbors and filter by similarity threshold
    sims = sims.head(params.neighbors_k)
    sims = sims[sims >= params.min_similarity]

    if sims.empty:
        return pd.DataFrame(columns=["movieId", "score", "support", "reasons"])

    neighbor_ids = sims.index.tolist()
    neighbor_ratings = ratings_df[ratings_df["userId"].isin(neighbor_ids)].copy()
    neighbor_ratings = neighbor_ratings[~neighbor_ratings["movieId"].isin(seen_movie_ids)]

    if neighbor_ratings.empty:
        return pd.DataFrame(columns=["movieId", "score", "support", "reasons"])

    # map each neighbor rating row -> similarity weight
    weight_map = sims.to_dict()
    neighbor_ratings["w"] = neighbor_ratings["userId"].map(weight_map).astype(float)

    # contribution for explainability
    neighbor_ratings["contrib"] = neighbor_ratings["w"] * neighbor_ratings["rating"]

    # weighted score per movie
    grouped = neighbor_ratings.groupby("movieId", as_index=False).agg(
        numerator=("contrib", "sum"),
        denom=("w", lambda s: float(np.sum(np.abs(s)))),
        support=("userId", "count"),
    )
    grouped["score"] = grouped["numerator"] / grouped["denom"].replace(0.0, np.nan)
    grouped = grouped.drop(columns=["numerator", "denom"]).dropna(subset=["score"])

    if params.min_raters > 1:
        grouped = grouped[grouped["support"] >= params.min_raters]

    # Explainability: top contributors per movie
    if params.explain:
        reasons_series = (
            neighbor_ratings.sort_values("contrib", ascending=False)
            .groupby("movieId", group_keys=False)
            .head(params.max_reasons)
            .groupby("movieId")
            .apply(
                lambda g: [
                    {
                        "type": "user_user",
                        "neighbor_userId": int(r["userId"]),
                        "similarity": float(r["w"]),
                        "neighbor_rating": float(r["rating"]),
                        "contribution": float(r["contrib"]),
                    }
                    for _, r in g.iterrows()
                ]
            )
        )

        grouped = grouped.merge(
            reasons_series.rename("reasons").reset_index(),
            on="movieId",
            how="left",
        )
    else:
        grouped["reasons"] = None

    return (
        grouped[["movieId", "score", "support", "reasons"]]
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


def _item_item_scores(
    user_id: int,
    ratings_df: pd.DataFrame,
    item_item_sim: pd.DataFrame,
    params: RecommendParams,
) -> pd.DataFrame:
    """
    Item-based CF score for unseen movie m using the user's rated items Iu:

        score(m) = sum(sim(m, i) * r(u, i)) / sum(|sim(m, i)|)

    Returns a DataFrame with columns:
      - movieId
      - score
      - support

    (Explainability for item-item can be extended later.)
    """
    user_hist = ratings_df.loc[ratings_df["userId"] == user_id, ["movieId", "rating"]].copy()
    if user_hist.empty:
        return pd.DataFrame(columns=["movieId", "score", "support"])

    seen_movie_ids = set(user_hist["movieId"].tolist())

    candidates = [mid for mid in item_item_sim.index.tolist() if mid not in seen_movie_ids]
    if not candidates:
        return pd.DataFrame(columns=["movieId", "score", "support"])

    rated_items = user_hist["movieId"].tolist()
    rated_r = user_hist.set_index("movieId")["rating"]

    rated_items = [i for i in rated_items if i in item_item_sim.columns]
    if not rated_items:
        return pd.DataFrame(columns=["movieId", "score", "support"])

    sim_sub = item_item_sim.loc[candidates, rated_items]

    if params.neighbors_k:
        sim_sub = sim_sub.apply(lambda row: row.nlargest(params.neighbors_k), axis=1).fillna(0.0)

    sim_sub = sim_sub.where(sim_sub >= params.min_similarity, other=0.0)

    r_vec = rated_r.loc[rated_items].astype(float)
    numerator = sim_sub.mul(r_vec, axis=1).sum(axis=1)
    denom = sim_sub.abs().sum(axis=1).replace(0.0, np.nan)
    score = numerator / denom

    out = pd.DataFrame(
        {
            "movieId": score.index.astype(int),
            "score": score.values.astype(float),
            "support": (sim_sub != 0.0).sum(axis=1).values.astype(int),
        }
    ).dropna(subset=["score"])

    if params.min_raters > 1:
        out = out[out["support"] >= params.min_raters]

    return out.sort_values("score", ascending=False).reset_index(drop=True)


def recommend_for_user(
    user_id: int,
    movies_df: Optional[pd.DataFrame],
    ratings_df: pd.DataFrame,
    user_user_sim: pd.DataFrame,
    item_item_sim: Optional[pd.DataFrame] = None,
    params: Optional[RecommendParams] = None,
) -> pd.DataFrame:
    """
    Compute top-N movie recommendations for a user using
    user-user, item-item, or hybrid CF.

    This function belongs to the pure *Domain / Code Layer*:
    - It performs in-memory numerical and tabular computations only.
    - It does NOT read/write files, touch databases, or log.

    Parameters
    ----------
    user_id:
        Target user identifier.

    movies_df:
        Optional metadata DataFrame with at least `movieId` column.
        If present and has a `title` column, titles are merged into
        the result; otherwise it is ignored.

    ratings_df:
        Long-form ratings DataFrame with columns:
        - userId
        - movieId
        - rating

    user_user_sim:
        User-user similarity matrix (square DataFrame) indexed
        and columned by userId.

    item_item_sim:
        Optional item-item similarity matrix (square DataFrame)
        indexed and columned by movieId. Required when alpha < 1.0.

    params:
        Recommendation hyperparameters. If None, defaults are used.

    Returns
    -------
    pd.DataFrame
        Sorted by score (descending). Columns:
        - movieId
        - score
        - support
        - reasons (None unless params.explain=True AND supported)
        - (optional) title if provided in movies_df
    """
    params = params or RecommendParams()
    _validate_inputs(user_id, ratings_df, user_user_sim, item_item_sim, params)

    # Pure user-user CF scores
    uu = _user_user_scores(user_id, ratings_df, user_user_sim, params)

    if params.alpha >= 1.0:
        combined = uu
    else:
        # Pure item-item CF scores
        ii = _item_item_scores(user_id, ratings_df, item_item_sim, params)

        # Make schemas compatible (so merges are clean)
        if "reasons" not in ii.columns:
            ii["reasons"] = None

        merged = pd.merge(
            uu,
            ii,
            on="movieId",
            how="outer",
            suffixes=("_uu", "_ii"),
        ).fillna(0.0)

        merged["score"] = (
            params.alpha * merged["score_uu"] + (1.0 - params.alpha) * merged["score_ii"]
        )
        merged["support"] = merged["support_uu"].astype(int) + merged["support_ii"].astype(int)

        # For now, in hybrid we keep reasons from user-user only
        merged["reasons"] = merged.get("reasons_uu", None)

        combined = merged[["movieId", "score", "support", "reasons"]].sort_values(
            "score",
            ascending=False,
        )

    combined = combined.reset_index(drop=True).head(params.top_k)

    # Optional enrichment with titles if provided (still in-memory, no I/O)
    if movies_df is not None and "movieId" in movies_df.columns:
        cols = ["movieId"]
        if "title" in movies_df.columns:
            cols.append("title")
        combined = combined.merge(
            movies_df[cols].drop_duplicates("movieId"),
            on="movieId",
            how="left",
        )

    # Final stable ordering: score desc, movieId asc
    combined = combined.sort_values(
        ["score", "movieId"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return combined
