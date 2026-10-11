"""Inference interventions at the full fusion model's selected threshold."""

import numpy as np

from .decoding import decode


def greedy(rows, threshold):
    kept, occupied = set(), set()
    for start, end, typ, probability in sorted(rows, key=lambda row: (-row[3], row[0], row[1], row[2])):
        if min(1 - 1e-6, max(1e-6, probability)) <= threshold or occupied.intersection(range(start, end)):
            continue
        kept.add((start, end, typ))
        occupied.update(range(start, end))
    return kept


def ablate(spans, baseline, logits, types, threshold):
    z = np.asarray(logits, dtype=np.float64)
    if not len(spans):
        pool = np.empty((0, len(types)), dtype=np.float64)
    else:
        weights = np.exp(z - z.max(axis=1, keepdims=True))
        pool = (weights / weights.sum(axis=1, keepdims=True))[:, 1:]
    lattice, allowed = np.zeros_like(pool), np.zeros_like(pool, dtype=bool)
    lookup = {tuple(span): i for i, span in enumerate(spans)}
    for start, end, typ, probability in baseline["lattice"]:
        if (start, end) in lookup:
            i, j = lookup[(start, end)], types.index(typ)
            lattice[i, j], allowed[i, j] = probability, True
    fused = (pool + lattice) / 2

    def best_rows(scores, eligible=None):
        if not len(spans):
            return []
        values = scores if eligible is None else np.where(eligible, scores, -1.0)
        chosen = np.argmax(values, axis=1)
        return [
            [int(start), int(end), types[int(typ)], float(scores[i, typ])]
            for i, ((start, end), typ) in enumerate(zip(spans, chosen))
        ]

    full = best_rows(fused)
    bio_bounds = {(row["s"], row["e"]) for row in baseline["bio"]}
    candidates = dict(
        FUSION=full,
        POOL_ONLY_FIXED=best_rows(pool),
        LATTICE_ONLY_FIXED=best_rows(lattice),
        BIO_BOUNDARIES_FIXED=[row for row in full if tuple(row[:2]) in bio_bounds],
        PROPOSAL_TYPES_FIXED=best_rows(fused, allowed),
        GREEDY_FIXED=full,
    )
    return {
        name: greedy(rows, threshold) if name == "GREEDY_FIXED" else decode(rows, threshold)
        for name, rows in candidates.items()
    }
