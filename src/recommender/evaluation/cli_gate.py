import sys
import argparse
import pandas as pd

from .runner import evaluate_baselines
from .model_runner import evaluate_model_on_holdout
from .protocol import apply_baseline_gate


# Default acceptance contract:
# model_recall must be at least popularity_recall * factor (+ margin)
DEFAULT_ACCEPTANCE_FACTOR = 1.10
DEFAULT_MARGIN = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation gate for recommender system")

    # --- Model enforcement control ---
    # Enforced by default (product/CI behavior), can be disabled explicitly for dev.
    parser.add_argument(
        "--no-enforce-model",
        action="store_true",
        help="Disable model acceptance enforcement (dev-only).",
    )

    # --- Acceptance contract knobs ---
    parser.add_argument(
        "--acceptance-factor",
        type=float,
        default=DEFAULT_ACCEPTANCE_FACTOR,
        help=(
            "Acceptance factor over popularity baseline. "
            "Requirement: model_recall >= popularity_recall * factor + margin"
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="Additional required margin added after factor scaling (recall@k).",
    )

    # (Optional) allow changing k/seed if you ever want it later, but keep stable defaults now.
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-K for evaluation metrics (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic evaluation (default: 42).",
    )

    args = parser.parse_args()

    ratings = pd.read_csv("tests/fixtures/ratings_small.csv")
    movies = pd.read_csv("data/movies.csv")

    # ---- Baseline gate (always enforced) ----
    baseline_metrics = evaluate_baselines(
        ratings=ratings,
        top_k=args.k,
        seed=args.seed,
    )
    baseline_result = apply_baseline_gate(baseline_metrics)

    print("Baseline Evaluation Gate Result:")
    print(baseline_result)
    print(baseline_result.details)

    if not baseline_result.passed:
        sys.exit(1)

    # ---- Model evaluation (always computed; enforcement optional) ----
    model_metrics = evaluate_model_on_holdout(
        ratings=ratings,
        movies=movies,
        top_k=args.k,
        seed=args.seed,
    )

    print("\nModel Metrics:")
    print(model_metrics)

    model_recall = float(model_metrics["model"]["recall@k"])
    popularity_recall = float(baseline_metrics["popularity"]["recall@k"])

    required = (popularity_recall * float(args.acceptance_factor)) + float(args.margin)

    print(
        "\nModel Acceptance Gate: "
        f"model_recall={model_recall:.6f}, "
        f"required>={required:.6f} "
        f"(factor={float(args.acceptance_factor):.2f}x, margin={float(args.margin):.6f})"
    )

    if args.no_enforce_model:
        print("Model enforcement disabled (dev-only).")
        return

    if model_recall < required:
        print("Model Acceptance gate FAILED")
        sys.exit(1)

    print("Model Acceptance gate PASSED")


if __name__ == "__main__":
    main()
