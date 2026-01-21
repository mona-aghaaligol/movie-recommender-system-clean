from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.recommender.models.svd_model import MatrixFactorization


DEFAULT_EMBEDDING_DIM: int = 32
DEFAULT_EPOCHS: int = 5
DEFAULT_LR: float = 0.01
DEFAULT_BATCH_SIZE: int = 1024
DEFAULT_SAMPLE_SIZE: int = 5000
DEFAULT_RANDOM_SEED: int = 42


@dataclass(frozen=True)
class SVDEvaluationResult:
    """
    Result of training and evaluating a MatrixFactorization (SVD-style) model.

    Attributes
    ----------
    mae : float
        Mean Absolute Error between actual and predicted ratings.
    rmse : float
        Root Mean Squared Error between actual and predicted ratings.
    n_observations : int
        Number of rating observations actually used in the evaluation.
    embedding_dim : int
        Latent dimension used for user/item embeddings.
    epochs : int
        Number of training epochs used during model training.
    lr : float
        Learning rate used by the optimizer.
    batch_size : int
        Batch size used for mini-batch training.
    sample_size : int
        Target number of samples for evaluation (clamped by dataset size).
    """
    mae: float
    rmse: float
    n_observations: int
    embedding_dim: int
    epochs: int
    lr: float
    batch_size: int
    sample_size: int


def evaluate_svd_detailed(
    ratings_df: pd.DataFrame,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_state: Optional[int] = DEFAULT_RANDOM_SEED,
) -> SVDEvaluationResult:
    """
    Train a MatrixFactorization model on the provided ratings DataFrame
    and evaluate it on a random sample of (userId, movieId, rating) rows.

    This is a pure domain/code-layer function:
    - It does not read from or write to files or databases.
    - It does not print or log anything.
    - It only consumes in-memory data structures and returns a typed result object.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        DataFrame with at least the following columns:
        - "userId"
        - "movieId"
        - "rating"
    embedding_dim : int, default=DEFAULT_EMBEDDING_DIM
        Latent dimension for user/item embeddings.
    epochs : int, default=DEFAULT_EPOCHS
        Number of training epochs.
    lr : float, default=DEFAULT_LR
        Learning rate for the optimizer used inside MatrixFactorization.
    batch_size : int, default=DEFAULT_BATCH_SIZE
        Batch size for the DataLoader inside the model.
    sample_size : int, default=DEFAULT_SAMPLE_SIZE
        Number of samples to use for evaluation. If larger than the dataset
        size, the full dataset is used.
    random_state : Optional[int], default=DEFAULT_RANDOM_SEED
        Seed for sampling reproducibility. If None, sampling will be
        non-deterministic.

    Returns
    -------
    SVDEvaluationResult
        Structured evaluation result including both metrics and hyperparameters.

    Raises
    ------
    RuntimeError
        If no valid predictions are produced during evaluation.
    ValueError
        If ratings_df is empty.
    """
    if ratings_df.empty:
        raise ValueError("ratings_df is empty; cannot train/evaluate SVD model.")

    # Clamp sample size to the available number of rows
    n_samples = min(sample_size, len(ratings_df))

    # Initialize and train the model (no external I/O here)
    model = MatrixFactorization(
        ratings_df=ratings_df,
        embedding_dim=embedding_dim,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    # Keep verbose=False so evaluation remains quiet and test-friendly.
    model.train(verbose=False)

    # Use a deterministic sample for evaluation if random_state is provided.
    test_df = ratings_df.sample(
        n=n_samples,
        random_state=random_state,
    )

    actual: list[float] = []
    predicted: list[float] = []

    for _, row in test_df.iterrows():
        user_id = row["userId"]
        movie_id = row["movieId"]

        try:
            pred = model.predict(user_id, movie_id)
        except ValueError:
            # Unknown user/movie (should not normally happen after training),
            # but we guard just in case.
            continue

        actual.append(float(row["rating"]))
        predicted.append(float(pred))

    if not actual:
        raise RuntimeError("No valid predictions were produced during SVD evaluation.")

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    return SVDEvaluationResult(
        mae=mae,
        rmse=rmse,
        n_observations=len(actual),
        embedding_dim=embedding_dim,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        sample_size=n_samples,
    )


def evaluate_svd(
    ratings_df: pd.DataFrame,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_state: Optional[int] = DEFAULT_RANDOM_SEED,
) -> Dict[str, float]:
    """
    Backwards-compatible wrapper around `evaluate_svd_detailed`.

    This helper preserves the original public API by returning a simple
    metrics dictionary of the form:
        {"mae": <float>, "rmse": <float>}

    Parameters
    ----------
    ratings_df : pd.DataFrame
        See `evaluate_svd_detailed`.
    embedding_dim : int, default=DEFAULT_EMBEDDING_DIM
        See `evaluate_svd_detailed`.
    epochs : int, default=DEFAULT_EPOCHS
        See `evaluate_svd_detailed`.
    lr : float, default=DEFAULT_LR
        See `evaluate_svd_detailed`.
    batch_size : int, default=DEFAULT_BATCH_SIZE
        See `evaluate_svd_detailed`.
    sample_size : int, default=DEFAULT_SAMPLE_SIZE
        See `evaluate_svd_detailed`.
    random_state : Optional[int], default=DEFAULT_RANDOM_SEED
        See `evaluate_svd_detailed`.

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary containing:
        - "mae": Mean Absolute Error
        - "rmse": Root Mean Squared Error
    """
    result = evaluate_svd_detailed(
        ratings_df=ratings_df,
        embedding_dim=embedding_dim,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        sample_size=sample_size,
        random_state=random_state,
    )
    return {"mae": result.mae, "rmse": result.rmse}
