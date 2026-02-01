from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .domain.predict_cf import predict_rating


DEFAULT_RANDOM_SEED: int = 42


@dataclass(frozen=True)
class CFEvaluationResult:
    """
    Result of evaluating a Collaborative Filtering (CF) model.

    Attributes
    ----------
    mae : float
        Mean Absolute Error between actual and predicted ratings.
    rmse : float
        Root Mean Squared Error between actual and predicted ratings.
    n_observations : int
        Number of rating observations actually used in the evaluation.
    """
    mae: float
    rmse: float
    n_observations: int


def _compute_mae_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute MAE and RMSE given two aligned arrays of actual and predicted values.

    Parameters
    ----------
    actual : np.ndarray
        Ground-truth target values.
    predicted : np.ndarray
        Model predictions.

    Returns
    -------
    mae : float
        Mean Absolute Error.
    rmse : float
        Root Mean Squared Error.
    """
    if actual.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch between actual {actual.shape} and predicted {predicted.shape}."
        )

    mae = float(np.mean(np.abs(predicted - actual)))
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    return mae, rmse


def evaluate_cf_model_detailed(
    ratings_df: pd.DataFrame,
    user_movie_matrix: pd.DataFrame,
    similarity_df: pd.DataFrame,
    random_state: Optional[int] = DEFAULT_RANDOM_SEED,
) -> CFEvaluationResult:
    """
    Evaluate a CF model using a simple leave-one-out-per-user strategy.

    This is a pure domain/code-layer function:
    - It does not read from or write to files or databases.
    - It does not print or log anything.
    - It only consumes in-memory data structures and returns a typed result object.

    For each user with at least two ratings:
    - Randomly select a single rated movie.
    - Attempt to predict its rating using `predict_rating`.
    - Collect (actual, predicted) pairs for all users where prediction is possible.
    - Compute MAE and RMSE over all collected pairs.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        DataFrame containing user ratings. Must include at least:
        - "userId"
        - "movieId"
        - "rating"
    user_movie_matrix : pd.DataFrame
        User-item rating matrix where:
        - Index: userId
        - Columns: movieId
        - Values: rating (float) or NaN.
    similarity_df : pd.DataFrame
        Item-item similarity matrix where:
        - Index: movieId
        - Columns: movieId
        - Values: similarity score (float).
    random_state : Optional[int], default=DEFAULT_RANDOM_SEED
        Seed used to control the randomness of movie selection per user.
        If None, a non-deterministic random generator will be used.

    Returns
    -------
    CFEvaluationResult
        A result object containing MAE, RMSE, and number of observations.

    Raises
    ------
    ValueError
        If no valid rating observations are available for evaluation.
    """
    if ratings_df.empty:
        raise ValueError("ratings_df is empty; cannot evaluate CF model.")

    # Build a dense user-movie matrix from the ratings DataFrame.
    full_matrix = ratings_df.pivot(
        index="userId",
        columns="movieId",
        values="rating",
    )

    actual_ratings: list[float] = []
    predicted_ratings: list[float] = []

    # Use a dedicated random generator for reproducibility and testability.
    rng = (
        np.random.default_rng(seed=random_state)
        if random_state is not None
        else np.random.default_rng()
    )

    for user_id in full_matrix.index:
        # All rated movies for this user
        user_ratings = full_matrix.loc[user_id].dropna()
        if len(user_ratings) < 2:
            # With fewer than 2 ratings, leave-one-out is not meaningful
            continue

        # Randomly select one rated movie to "hold out"
        movie_id = rng.choice(user_ratings.index.to_numpy())
        true_rating = float(user_ratings[movie_id])

        # If the movie is not present in the similarity matrix,
        # we cannot compute a CF prediction for it.
        if movie_id not in similarity_df.index:
            continue

        # Domain-level prediction call: this function is expected to be pure
        # with respect to the given matrices (no external I/O).
        predicted = predict_rating(
            user_id=user_id,
            movie_id=movie_id,
            user_movie_matrix=user_movie_matrix,
            similarity_df=similarity_df,
        )

        # Skip cases where the predictor cannot produce a rating.
        if pd.isna(predicted):
            continue

        actual_ratings.append(true_rating)
        predicted_ratings.append(float(predicted))

    if not actual_ratings:
        raise ValueError("No valid rating observations were found for CF evaluation.")

    actual_arr = np.asarray(actual_ratings, dtype=float)
    predicted_arr = np.asarray(predicted_ratings, dtype=float)

    mae, rmse = _compute_mae_rmse(actual_arr, predicted_arr)

    return CFEvaluationResult(
        mae=mae,
        rmse=rmse,
        n_observations=len(actual_ratings),
    )


def evaluate_cf_model(
    ratings_df: pd.DataFrame,
    user_movie_matrix: pd.DataFrame,
    similarity_df: pd.DataFrame,
    random_state: Optional[int] = DEFAULT_RANDOM_SEED,
) -> Tuple[float, float]:
    """
    Backwards-compatible wrapper for CF evaluation.

    This function preserves the original public API by returning a (mae, rmse) tuple,
    while delegating the core logic to `evaluate_cf_model_detailed`.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        See `evaluate_cf_model_detailed`.
    user_movie_matrix : pd.DataFrame
        See `evaluate_cf_model_detailed`.
    similarity_df : pd.DataFrame
        See `evaluate_cf_model_detailed`.
    random_state : Optional[int], default=DEFAULT_RANDOM_SEED
        See `evaluate_cf_model_detailed`.

    Returns
    -------
    mae : float
        Mean Absolute Error of the CF model.
    rmse : float
        Root Mean Squared Error of the CF model.
    """
    result = evaluate_cf_model_detailed(
        ratings_df=ratings_df,
        user_movie_matrix=user_movie_matrix,
        similarity_df=similarity_df,
        random_state=random_state,
    )
    return result.mae, result.rmse
