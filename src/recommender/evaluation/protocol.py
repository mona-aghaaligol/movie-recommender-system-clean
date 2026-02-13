from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, List


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / float(k)


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / float(len(relevant))


@dataclass(frozen=True)
class EvaluationGateResult:
    passed: bool
    reason: str
    details: Dict


def apply_baseline_gate(
    metrics: Dict,
    *,
    min_recall_at_k: float = 0.01,
) -> EvaluationGateResult:
    """
    Contract v1 (CI-safe):
      1) popularity must beat random (sanity check)
      2) popularity recall@k must be >= min_recall_at_k

    This is a baseline-only gate. In Step 3+ we will add model-vs-baseline gating.
    """
    try:
        k = metrics["meta"]["k"]
        pop_recall = float(metrics["popularity"]["recall@k"])
        rnd_recall = float(metrics["random"]["recall@k"])
        users = int(metrics["meta"]["users_evaluated"])
    except Exception as e:
        return EvaluationGateResult(
            passed=False,
            reason=f"Invalid metrics schema: {e}",
            details={"metrics": metrics},
        )

    if users <= 0:
        return EvaluationGateResult(
            passed=False,
            reason="No eligible users were evaluated (users_evaluated <= 0).",
            details={"users_evaluated": users, "k": k},
        )

    if pop_recall <= rnd_recall:
        return EvaluationGateResult(
            passed=False,
            reason="Sanity check failed: popularity recall@k is not better than random.",
            details={
                "k": k,
                "users_evaluated": users,
                "popularity_recall@k": pop_recall,
                "random_recall@k": rnd_recall,
            },
        )

    if pop_recall < min_recall_at_k:
        return EvaluationGateResult(
            passed=False,
            reason=f"Quality threshold failed: popularity recall@k < {min_recall_at_k}.",
            details={
                "k": k,
                "users_evaluated": users,
                "popularity_recall@k": pop_recall,
                "min_recall_at_k": min_recall_at_k,
            },
        )

    return EvaluationGateResult(
        passed=True,
        reason="Baseline gate passed.",
        details={
            "k": k,
            "users_evaluated": users,
            "popularity_recall@k": pop_recall,
            "random_recall@k": rnd_recall,
            "min_recall_at_k": min_recall_at_k,
        },
    )
