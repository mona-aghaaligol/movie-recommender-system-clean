import sys
import pandas as pd

from .runner import evaluate_baselines
from .protocol import apply_baseline_gate


def main() -> None:
    ratings = pd.read_csv("tests/fixtures/ratings_small.csv")

    metrics = evaluate_baselines(
        ratings=ratings,
        top_k=10,
        seed=42,
    )

    result = apply_baseline_gate(metrics)

    print("Evaluation Gate Result:")
    print(result)

    if not result.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

