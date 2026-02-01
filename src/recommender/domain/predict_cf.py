from __future__ import annotations

from typing import Hashable

import numpy as np
import pandas as pd


def predict_rating(
    user_id: Hashable,
    movie_id: Hashable,
    user_movie_matrix: pd.DataFrame,
    similarity_df: pd.DataFrame,
) -> float:
    """
    Predict a single rating for (user_id, movie_id) using
    item-based collaborative filtering with a user-mean baseline.

    This function belongs to the pure *Domain / Code Layer*:
    it performs in-memory computations only and has no side effects
    (no file I/O, no database access, no logging, no printing).

    Parameters
    ----------
    user_id:
        Identifier of the target user. Must exist in `user_movie_matrix.index`.

    movie_id:
        Identifier of the target movie/item. Must exist in `similarity_df.index`
        and `similarity_df.columns`.

    user_movie_matrix:
        User–item rating matrix (rows: users, columns: movies/items).
        Entries are numeric ratings or NaN for missing ratings.

    similarity_df:
        Item–item similarity matrix aligned with the columns of
        `user_movie_matrix`. Both its index and columns should be movie IDs.

    Returns
    -------
    float
        Predicted rating as a float.
        Returns `np.nan` if the user has no rated movies (no signal).

    Raises
    ------
    KeyError
        If `user_id` is not present in `user_movie_matrix.index` or
        `movie_id` is not present in `similarity_df`.
    """
    # 1️⃣ Ratings of the target user
    user_ratings = user_movie_matrix.loc[user_id]

    # 2️⃣ Movies rated by this user (S_u)
    rated_mask = user_ratings.notna()
    if not rated_mask.any():
        # The user has no rating history → no signal at all
        return float("nan")

    rated_movies = user_ratings.index[rated_mask]

    # 3️⃣ User mean over rated movies
    user_mean = float(user_ratings[rated_movies].mean())

    # 4️⃣ Similarities between target movie and movies rated by the user
    similarities = similarity_df.loc[movie_id, rated_movies]

    # 5️⃣ Deviation from user mean for each rated movie
    rating_diffs = user_ratings[rated_movies] - user_mean

    # 6️⃣ Drop pairs where similarity or rating diff is NaN
    valid_mask = similarities.notna() & rating_diffs.notna()
    if not valid_mask.any():
        # No valid neighbor left → fall back to user mean
        return user_mean

    similarities = similarities[valid_mask].to_numpy()
    rating_diffs = rating_diffs[valid_mask].to_numpy()

    # 7️⃣ Weighted sum
    numerator = float(np.dot(similarities, rating_diffs))
    denominator = float(np.sum(np.abs(similarities)))

    # 8️⃣ If denominator is zero, prediction collapses to user mean
    if denominator == 0.0:
        return user_mean

    # 9️⃣ Final prediction
    return user_mean + numerator / denominator
