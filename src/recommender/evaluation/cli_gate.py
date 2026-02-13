
import sys
import argparse
import pandas as pd

from .runner import evaluate_baselines
from .model_runner import evaluate_model_on_holdout
from .protocol import apply_baseline_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation gate for recommender system")
    parser.add_argument(
        "--enforce-model",
        action="store_true",
        help="Fail if model does not beat popularity baseline",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Required margin over popularity baseline (recall@k)",
    )
    args = parser.parse_args()

    ratings = pd.read_csv("tests/fixtures/ratings_small.csv")
    movies = pd.read_csv("data/movies.csv")

    # ---- Baseline gate (always enforced) ----
    baseline_metrics = evaluate_baselines(
        ratings=ratings,
        top_k=10,
        seed=42,
    )
    baseline_result = apply_baseline_gate(baseline_metrics)

    print("Baseline Evaluation Gate Result:")
    print(baseline_result)
    print(baseline_result.details)

    if not baseline_result.passed:
        sys.exit(1)

    # ---- Model evaluation (optional enforcement) ----
    model_metrics = evaluate_model_on_holdout(
        ratings=ratings,
        movies=movies,
        top_k=10,
        seed=42,
    )

    print("\nModel Metrics:")
    print(model_metrics)

    if args.enforce_model:
        model_recall = model_metrics["model"]["recall@k"]
        baseline_recall = baseline_metrics["popularity"]["recall@k"]

        required = baseline_recall + args.margin

        print(
            f"\nModel-vs-Baseline Gate: "
            f"model_recall={model_recall:.6f}, "
            f"required>={required:.6f}"
        )

        if model_recall < required:
            print("Model-vs-Baseline gate FAILED")
            sys.exit(1)

        print("Model-vs-Baseline gate PASSED")


if __name__ == "__main__":
    main()

