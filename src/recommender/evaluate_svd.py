import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.recommender.models.svd_model import MatrixFactorization


def evaluate_svd(
    ratings_df: pd.DataFrame,
    embedding_dim: int = 32,
    epochs: int = 5,
    lr: float = 0.01,
    batch_size: int = 1024,
    sample_size: int = 5000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Train a MatrixFactorization model on the provided ratings DataFrame
    and evaluate it on a random sample of (userId, movieId, rating) rows.

    Parameters
    ----------
    ratings_df:
        DataFrame with at least columns: userId, movieId, rating.
    embedding_dim:
        Latent dimension for user/item embeddings.
    epochs:
        Number of training epochs.
    lr:
        Learning rate for Adam optimizer.
    batch_size:
        Batch size for DataLoader.
    sample_size:
        Number of samples to use for evaluation. If larger than the
        dataset size, the full dataset is used.
    random_state:
        Seed for sampling reproducibility.

    Returns
    -------
    metrics:
        Dictionary containing:
        - "mae": Mean Absolute Error
        - "rmse": Root Mean Squared Error
    """
    # Ensure we don't sample more rows than available
    n_samples = min(sample_size, len(ratings_df))

    # Create and train the model
    model = MatrixFactorization(
        ratings_df=ratings_df,
        embedding_dim=embedding_dim,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    # We keep verbose=False so this function is quiet by default
    model.train(verbose=False)

    # Sample a subset for evaluation
    test_df = ratings_df.sample(n_samples, random_state=random_state)

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
        raise RuntimeError("No valid predictions were produced during evaluation.")

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    return {"mae": mae, "rmse": rmse}


def main() -> None:
    """
    Script entrypoint: load ratings, train + evaluate SVD, and print metrics.
    """
    print("📥 Loading dataset...")

    # Compute path to data/ratings.csv relative to this file
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/ratings.csv")
    )
    ratings_df = pd.read_csv(data_path)

    print("🔍 Training and evaluating SVD model...")
    metrics: Dict[str, Any] = evaluate_svd(ratings_df)

    print("\n✅ SVD Evaluation Completed")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
