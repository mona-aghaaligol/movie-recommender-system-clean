from __future__ import annotations

from typing import Literal, Optional

import pandas as pd

AggFunc = Literal["mean", "sum", "median", "min", "max"]


def build_user_movie_matrix(
    ratings: pd.DataFrame,
    *,
    user_col: str = "userId",
    movie_col: str = "movieId",
    rating_col: str = "rating",
    aggfunc: AggFunc = "mean",
    fill_missing: Optional[float] = None,
    sort_index: bool = True,
) -> pd.DataFrame:
    """
    Build a user–movie rating matrix from a long-form ratings DataFrame.

    This function belongs to the pure *Domain / Code Layer*:
    it performs in-memory computations only and has no side effects
    (no file I/O, no database access, no logging, no printing).

    Parameters
    ----------
    ratings:
        Long-form DataFrame with at least three columns:
        - `user_col`: user identifier
        - `movie_col`: movie/item identifier
        - `rating_col`: numeric rating value

    user_col:
        Name of the column containing user IDs.

    movie_col:
        Name of the column containing movie IDs.

    rating_col:
        Name of the column containing rating values.

    aggfunc:
        Aggregation function to apply when a user has multiple ratings
        for the same movie. Typical choices: "mean", "sum", "median",
        "min", "max".

    fill_missing:
        If provided, all missing entries in the matrix are filled with
        this value. If None, missing entries remain as NaN (sparse
        representation), which is often preferred for similarity-based
        methods.

    sort_index:
        If True, sort both users (index) and movies (columns) to obtain
        a stable matrix layout across runs. This is helpful for
        reproducibility and for aligning matrices between components.

    Returns
    -------
    pd.DataFrame
        A pivoted matrix where:
            - rows   = users (unique values of `user_col`)
            - columns = movies/items (unique values of `movie_col`)
            - values = aggregated ratings

        The matrix dtype is `float32` to balance numeric precision and
        memory usage for large-scale recommendation tasks.

    Raises
    ------
    TypeError
        If `ratings` is not a pandas DataFrame.

    ValueError
        If required columns are missing from `ratings`.
    """
    if not isinstance(ratings, pd.DataFrame):
        raise TypeError(
            f"`ratings` must be a pandas DataFrame, got {type(ratings)!r} instead."
        )

    required_cols = {user_col, movie_col, rating_col}
    missing_cols = required_cols.difference(ratings.columns)
    if missing_cols:
        # Fail fast with a clear message about what is wrong.
        missing_str = ", ".join(sorted(missing_cols))
        expected_str = ", ".join(sorted(required_cols))
        raise ValueError(
            f"`ratings` is missing required columns: {missing_str}. "
            f"Expected at least: {expected_str}."
        )

    if ratings.empty:
        # For an empty input, return an empty matrix with a numeric dtype.
        # This keeps the function total and avoids surprising None returns.
        return pd.DataFrame(dtype="float32")

    matrix = ratings.pivot_table(
        index=user_col,
        columns=movie_col,
        values=rating_col,
        aggfunc=aggfunc,
    ).astype("float32")

    if fill_missing is not None:
        matrix = matrix.fillna(fill_missing)

    if sort_index:
        # Stable ordering: important if matrices are compared across runs,
        # or aligned with other components.
        matrix = matrix.sort_index(axis=0).sort_index(axis=1)

    return matrix
