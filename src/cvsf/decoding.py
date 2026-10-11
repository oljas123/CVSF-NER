"""Equal-weight score fusion and non-overlapping interval decoding."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def weighted_interval_decode(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    candidates = []
    threshold_logit = logit(threshold)
    for row in rows:
        prob = float(row.get("reranker_probability") or 0.0)
        weight = logit(prob) - threshold_logit
        if weight > 0:
            item = dict(row)
            item["_decode_weight"] = weight
            candidates.append(item)
    candidates.sort(key=lambda row: (int(row["end"]), int(row["start"]), -float(row["reranker_probability"])))
    n = len(candidates)
    if n == 0:
        return []
    previous = []
    ends = [int(row["end"]) for row in candidates]
    for i, row in enumerate(candidates):
        start = int(row["start"])
        j = i - 1
        while j >= 0 and ends[j] > start:
            j -= 1
        previous.append(j)
    dp = [0.0] * (n + 1)
    take = [False] * n
    for i, row in enumerate(candidates, start=1):
        include = float(row["_decode_weight"]) + dp[previous[i - 1] + 1]
        exclude = dp[i - 1]
        if include > exclude:
            dp[i] = include
            take[i - 1] = True
        else:
            dp[i] = exclude
    selected = []
    i = n - 1
    while i >= 0:
        if take[i] and float(candidates[i]["_decode_weight"]) + dp[previous[i] + 1] >= dp[i]:
            selected.append(candidates[i])
            i = previous[i]
        else:
            i -= 1
    selected.sort(key=lambda row: (int(row["start"]), int(row["end"])))
    for row in selected:
        row.pop("_decode_weight", None)
    return selected


def decode(scored_spans, threshold):
    """Return typed, half-open spans with maximum total threshold-adjusted log odds."""
    rows = [
        dict(start=int(s), end=int(e), pred_type=str(t), reranker_probability=float(p)) for s, e, t, p in scored_spans
    ]
    return {(r["start"], r["end"], r["pred_type"]) for r in weighted_interval_decode(rows, float(threshold))}


def fuse(spans, lattice, logits, types):
    """Score each boundary under POOL and the equal-weight fusion (FUSION)."""
    z = np.asarray(logits, dtype=np.float64)
    if z.shape != (len(spans), len(types) + 1):
        raise ValueError("POOL logits must contain NONE followed by the ordered entity types")
    if not len(spans):
        return {"POOL": [], "FUSION": []}
    weights = np.exp(z - z.max(-1, keepdims=True))
    probability = (weights / weights.sum(-1, keepdims=True))[:, 1:]
    sparse = np.zeros_like(probability)
    lookup = {tuple(span): i for i, span in enumerate(spans)}
    for start, end, typ, score in lattice:
        if (start, end) in lookup:
            sparse[lookup[(start, end)], types.index(typ)] = score
    result = {}
    for name, values in [("POOL", probability), ("FUSION", (probability + sparse) / 2)]:
        chosen = values.argmax(-1)
        result[name] = [
            [int(s), int(e), types[int(j)], float(values[i, j])] for i, ((s, e), j) in enumerate(zip(spans, chosen))
        ]
    return result
