"""Top-k ranking metrics for recommenders and search (numpy only).

Conventions
-----------
* A ranking is a list of item ids, best first.
* Relevance is a dict {item_id: gain}; gain > 0 (or >= relevant_threshold) counts as relevant.
* Queries (users) with no relevant items are EXCLUDED from recall/MAP/NDCG (undefined) and
  reported in `n_skipped`; never silently counted as zero.
"""
from __future__ import annotations

import math
from typing import Dict, Hashable, Iterable, List, Mapping, Sequence

import numpy as np

from .stats import bootstrap_ci


def _relevant_set(gains: Mapping[Hashable, float], threshold: float = 1.0) -> set:
    return {i for i, g in gains.items() if g >= threshold}


def precision_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    rel = set(relevant)
    top = list(ranked)[:k]
    return sum(1 for i in top if i in rel) / k


def recall_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    rel = set(relevant)
    if not rel:
        raise ValueError("recall undefined without relevant items")
    return sum(1 for i in list(ranked)[:k] if i in rel) / len(rel)


def reciprocal_rank(ranked: Sequence[Hashable], relevant: Iterable[Hashable]) -> float:
    rel = set(relevant)
    for pos, i in enumerate(ranked, start=1):
        if i in rel:
            return 1.0 / pos
    return 0.0


def average_precision(ranked: Sequence[Hashable], relevant: Iterable[Hashable],
                      k: int | None = None) -> float:
    """AP@k = sum_{i<=k} P@i * rel_i / min(|R|, k). With k=None uses the full list and |R|."""
    rel = set(relevant)
    if not rel:
        raise ValueError("AP undefined without relevant items")
    top = list(ranked) if k is None else list(ranked)[:k]
    hits, score = 0, 0.0
    for pos, i in enumerate(top, start=1):
        if i in rel:
            hits += 1
            score += hits / pos
    denom = len(rel) if k is None else min(len(rel), k)
    return score / denom


def dcg_at_k(gains_in_rank_order: Sequence[float], k: int, exponential: bool = False) -> float:
    g = np.asarray(list(gains_in_rank_order)[:k], dtype=float)
    if g.size == 0:
        return 0.0
    num = (2.0 ** g - 1.0) if exponential else g
    return float(np.sum(num / np.log2(np.arange(2, g.size + 2))))


def ndcg_at_k(ranked: Sequence[Hashable], gains: Mapping[Hashable, float], k: int,
              exponential: bool = False) -> float:
    """NDCG@k with the ideal ranking built from ALL known gains of the query."""
    actual = [gains.get(i, 0.0) for i in ranked]
    ideal = sorted(gains.values(), reverse=True)
    idcg = dcg_at_k(ideal, k, exponential)
    if idcg == 0:
        raise ValueError("NDCG undefined when the query has no positive gain")
    return dcg_at_k(actual, k, exponential) / idcg


def catalog_coverage(rankings: Mapping[Hashable, Sequence[Hashable]], catalog_size: int, k: int) -> float:
    """Share of the catalog that appears in at least one top-k list."""
    seen = set()
    for r in rankings.values():
        seen.update(list(r)[:k])
    return len(seen) / catalog_size


def evaluate_rankings(rankings: Mapping[Hashable, Sequence[Hashable]],
                      relevance: Mapping[Hashable, Mapping[Hashable, float]],
                      ks: Sequence[int] = (5, 10), relevant_threshold: float = 1.0,
                      n_boot: int = 2_000, seed: int = 42) -> dict:
    """Evaluate P@k, R@k, NDCG@k, MAP@k and MRR over queries, with bootstrap CIs over queries.

    Returns {'n_evaluated', 'n_skipped', 'metrics': {name: {'mean','ci_lo','ci_hi'}},
             'per_query': {name: np.ndarray}}."""
    per: Dict[str, List[float]] = {}

    def add(name: str, v: float) -> None:
        per.setdefault(name, []).append(v)

    skipped = 0
    for q, ranked in rankings.items():
        gains = relevance.get(q, {})
        rel = _relevant_set(gains, relevant_threshold)
        if not rel:
            skipped += 1
            continue
        add("MRR", reciprocal_rank(ranked, rel))
        for k in ks:
            add(f"P@{k}", precision_at_k(ranked, rel, k))
            add(f"R@{k}", recall_at_k(ranked, rel, k))
            add(f"MAP@{k}", average_precision(ranked, rel, k))
            add(f"NDCG@{k}", ndcg_at_k(ranked, gains, k))
    if not per:
        raise ValueError("no query with relevant items to evaluate")
    metrics = {}
    for name, vals in per.items():
        m, lo, hi = bootstrap_ci(vals, np.mean, n_boot=n_boot, seed=seed)
        metrics[name] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
    n_eval = len(next(iter(per.values())))
    return {"n_evaluated": n_eval, "n_skipped": skipped, "metrics": metrics,
            "per_query": {k: np.asarray(v) for k, v in per.items()}}
