import sys
import argparse
import pandas as pd

from .runner import evaluate_baselines
from .model_runner import evaluate_model_on_holdout
from .protocol import apply_baseline_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation gate for recommender system")

    # Default: enforce model-vs-baseline.
    # Dev-only escape hatch:
    parser.add_argument(
        "--no-enforce-model",
        action="store_true",
        help="Disable model-vs-baseline enforcement (dev-only).",
    )

    # Acceptance contract:
    # model_recall >= popularity_recall * factor
    parser.add_argument(
        "--model-recall-factor",
        type=float,
        default=1.10,
        help="Required factor over popularity recall@k (default: 1.10).",
    )

    # Keep existing knobs for future tuning if needed
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Additional absolute margin over required recall@k (default: 0.0).",
    )

    args = parser.parse_args()

    enforce_model = not args.no_enforce_model
    factor = float(args.model_recall_factor)
    margin = float(args.margin)

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

    # ---- Model evaluation ----
    model_metrics = evaluate_model_on_holdout(
        ratings=ratings,
        movies=movies,
        top_k=10,
        seed=42,
    )
    print("\nModel Metrics:")
    print(model_metrics)

    # ---- Acceptance contract (default enforced) ----
    pop_recall = float(baseline_metrics["popularity"]["recall@k"])
    model_recall = float(model_metrics["model"]["recall@k"])
    required = (pop_recall * factor) + margin

    print(
        f"\nModel Acceptance Gate: "
        f"model_recall={model_recall:.6f}, "
        f"required>={required:.6f} "
        f"(factor={factor:.2f}x, margin={margin:.6f})"
    )

    if enforce_model:
        if model_recall < required:
            print("Model Acceptance gate FAILED")
            sys.exit(1)
        print("Model Acceptance gate PASSED")
    else:
        print("Model enforcement disabled (dev-only).")


if __name__ == "__main__":
    main()

