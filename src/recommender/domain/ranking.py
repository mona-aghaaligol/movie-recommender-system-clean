from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Hashable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Domain-level configuration for ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankParams:
    """
    Immutable configuration for ranking recommendations.

    Parameters
    ----------
    top_k:
        Number of final items to return after ranking.

    diversity:
        If True, apply MMR-based diversity re-ranking on top of the
        relevance score ordering. If False, only use stable sorting
        by score/support/movieId.

    mmr_lambda:
        Trade-off parameter for MMR:
        - closer to 1.0 -> focus more on relevance (score).
        - closer to 0.0 -> focus more on diversity (penalize similarity).

    mmr_candidates:
        Number of top-by-score candidates to consider for MMR re-ranking.
        This is typically larger than top_k to have enough candidates
        to diversify, but bounded for performance.
    """

    top_k: int = 10
    diversity: bool = False
    mmr_lambda: float = 0.8  # closer to 1 => relevance, closer to 0 => diversity
    mmr_candidates: int = 200  # only rerank top-N by score for speed


_REQUIRED_COLUMNS = ("movieId", "score")


# ---------------------------------------------------------------------------
# Internal helpers (pure functions)
# ---------------------------------------------------------------------------


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the scored candidates DataFrame contains the required columns.

    Expected columns:
      - 'movieId'
      - 'score'
    """
    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"scored_df is missing required columns: {missing}. "
            f"Expected at least: {_REQUIRED_COLUMNS}"
        )
    return df


def _stable_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stable, deterministic ranking of candidates by score/support/movieId.

    Sort order:
      - score : descending
      - support : descending (if present, otherwise treated as 0)
      - movieId : ascending
    """
    if df.empty:
        return df.copy()

    df = _ensure_required_columns(df).copy()

    if "support" not in df.columns:
        df = df.assign(support=0)

    return df.sort_values(
        ["score", "support", "movieId"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _mmr_rerank(
    df: pd.DataFrame,
    item_item_sim: pd.DataFrame,
    params: RankParams,
) -> pd.DataFrame:
    """
    Apply MMR (Maximal Marginal Relevance) re-ranking to promote diversity.

    At a high level, MMR iteratively selects items that balance:
      - relevance (score)
      - dissimilarity to items already selected

    MMR objective (for each candidate i):
        argmax λ * score(i) - (1 - λ) * max_{j in selected} sim(i, j)

    Requirements
    ------------
    - `df` must contain at least 'movieId' and 'score' columns.
    - `item_item_sim` must be a square item-item similarity matrix where:
        - index: movieId
        - columns: movieId
        - entries: similarity scores (higher = more similar)

    Notes
    -----
    - Only the top `mmr_candidates` items by score are considered for MMR.
    - Items missing in the similarity matrix are kept but not diversified.
    """
    if df.empty:
        return df.copy()

    df = _ensure_required_columns(df)

    # Make sure we consider enough candidates for diversification.
    mmr_candidates = max(int(params.top_k), int(params.mmr_candidates))
    candidates = df.head(mmr_candidates).copy()
    rest = df.iloc[mmr_candidates:].copy()

    # Only keep items present in sim matrix for diversity computation
    present_mask = candidates["movieId"].isin(item_item_sim.index)
    candidates_present = candidates[present_mask].reset_index(drop=True)
    candidates_missing = candidates[~present_mask].reset_index(drop=True)

    # If not enough items are present in the similarity matrix,
    # just fall back to stable ranking.
    if len(candidates_present) <= 1:
        out = pd.concat(
            [candidates_present, candidates_missing, rest],
            ignore_index=True,
        )
        return _stable_rank(out)

    # Precompute scores as a dict for fast lookup
    scores = candidates_present.set_index("movieId")["score"].to_dict()

    selected: list[Hashable] = []
    selected_set: set[Hashable] = set()

    # Pick the best-by-score item first (candidates_present is already sorted)
    first_movie_id = candidates_present.iloc[0]["movieId"]
    selected.append(first_movie_id)
    selected_set.add(first_movie_id)

    # Remaining items (by movieId)
    remaining = [
        mid for mid in candidates_present["movieId"].tolist()
        if mid != first_movie_id
    ]

    lam = float(params.mmr_lambda)
    # Clamp lambda to [0, 1] to avoid misconfiguration
    lam = max(0.0, min(1.0, lam))

    while remaining and len(selected) < params.top_k:
        best_mid: Optional[Hashable] = None
        best_val = -np.inf

        for mid in remaining:
            rel = float(scores.get(mid, 0.0))

            # diversity penalty: similarity to already-selected set
            sims: list[float] = []
            if mid in item_item_sim.index:
                for sid in selected:
                    if sid in item_item_sim.columns:
                        sims.append(float(item_item_sim.loc[mid, sid]))
            penalty = max(sims) if sims else 0.0

            val = lam * rel - (1.0 - lam) * penalty
            if val > best_val:
                best_val = val
                best_mid = mid

        if best_mid is None:
            # Should not happen in normal cases, but guard just in case.
            break

        selected.append(best_mid)
        selected_set.add(best_mid)
        remaining.remove(best_mid)

    # Build reranked list: selected first, then remaining by stable ranking
    selected_df = candidates_present[
        candidates_present["movieId"].isin(selected_set)
    ].copy()
    selected_df["__order"] = selected_df["movieId"].apply(
        lambda mid: selected.index(mid)
    )
    selected_df = selected_df.sort_values("__order").drop(columns="__order")

    leftover_df = candidates_present[
        ~candidates_present["movieId"].isin(selected_set)
    ].copy()
    leftover_df = _stable_rank(leftover_df)

    out = pd.concat(
        [selected_df, leftover_df, candidates_missing, rest],
        ignore_index=True,
    )
    out = _stable_rank(out)
    return out


# ---------------------------------------------------------------------------
# Public Domain API
# ---------------------------------------------------------------------------


def rank_recommendations(
    scored_df: pd.DataFrame,
    *,
    params: RankParams,
    item_item_sim: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Rank a set of scored candidate items into a final top-k list.

    Parameters
    ----------
    scored_df:
        DataFrame with at least the following columns:
          - 'movieId' : item identifier (must be hashable, typically int/str)
          - 'score'   : relevance score (higher = better)
        Optionally can contain:
          - 'support' : tie-breaker (e.g., number of ratings/users)

    params:
        Immutable configuration for ranking and diversity (top_k, etc.).

    item_item_sim:
        Optional item-item similarity matrix used for MMR diversity.
        Required if `params.diversity` is True.
        Expected format:
          - index: movieId
          - columns: movieId
          - values: similarity scores.

    Returns
    -------
    ranked_df:
        DataFrame of the top `params.top_k` items, ranked and reset_index-ed.
        The original columns are preserved.
    """
    if scored_df.empty:
        # Return an empty DataFrame with the same columns
        return scored_df.copy()

    # Validate required columns but don't mutate the caller's DataFrame
    scored_df = _ensure_required_columns(scored_df).copy()

    ranked = _stable_rank(scored_df)

    if params.diversity:
        if item_item_sim is None:
            raise ValueError(
                "rank_recommendations called with diversity=True but "
                "item_item_sim is None. Provide an item-item similarity "
                "matrix or disable diversity."
            )
        ranked = _mmr_rerank(ranked, item_item_sim, params)

    return ranked.head(params.top_k).reset_index(drop=True)
