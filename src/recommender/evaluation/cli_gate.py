import sys
import pandas as pd

from .runner import evaluate_baselines
from .model_runner import evaluate_model_on_holdout
from .protocol import apply_baseline_gate


def main() -> None:
    ratings = pd.read_csv("tests/fixtures/ratings_small.csv")
    movies = pd.read_csv("data/movies.csv")

    baseline_metrics = evaluate_baselines(
        ratings=ratings,
        top_k=10,
        seed=42,
    )
    baseline_result = apply_baseline_gate(baseline_metrics)

    print("Baseline Evaluation Gate Result:")
    print(baseline_result)
    print(baseline_result.details)

    # Always compute model metrics for visibility, but do not fail CI yet.
    model_metrics = evaluate_model_on_holdout(
        ratings=ratings,
        movies=movies,
        top_k=10,
        seed=42,
    )
    print("\nModel Metrics (informational):")
    print(model_metrics)

    # CI FAIL only if baseline gate fails (stable, deterministic).
    if not baseline_result.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

