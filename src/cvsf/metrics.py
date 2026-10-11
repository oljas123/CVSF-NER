"""Exact typed-span micro precision, recall and F1."""

from __future__ import annotations

from collections import Counter

import numpy as np


def entities(tags, strict=False):
    """Convert BIO tags to half-open spans; an orphan predicted I starts a span."""
    result = []
    active = None
    for i, tag in enumerate(list(tags) + ["O"]):
        tag = str(tag)
        if tag == "O":
            prefix, typ = "O", None
        elif len(tag) > 2 and tag[0] in "BI" and tag[1] == "-":
            prefix, typ = tag[0], tag[2:]
        else:
            raise ValueError(f"Unsupported BIO tag: {tag}")
        same_inside = prefix == "I" and active is not None and active[1] == typ
        if strict and prefix == "I" and not same_inside:
            raise ValueError(f"Orphan gold I at {i}: {tag}")
        if active is not None and not same_inside:
            result.append((active[0], i, active[1]))
            active = None
        if prefix == "B" or (prefix == "I" and not same_inside):
            active = (i, typ)
    return result


def metric(tp, pred, gold):
    return {
        "tp": int(tp),
        "fp": int(pred - tp),
        "fn": int(gold - tp),
        "pred": int(pred),
        "gold": int(gold),
        "precision": tp / pred if pred else 0.0,
        "recall": tp / gold if gold else 0.0,
        "f1": 2 * tp / (pred + gold) if pred + gold else 0.0,
    }


def extract_entities(tags, tokens=None):
    result = [dict(start=s, end=e, type=t) for s, e, t in entities(tags)]
    if tokens is not None:
        for row in result:
            row["text"] = " ".join(tokens[row["start"] : row["end"]])
    return result


def evaluate(predictions, gold, indices=None):
    if len(predictions) != len(gold):
        raise ValueError("Prediction and gold sentence counts differ")
    indices = range(len(gold)) if indices is None else indices
    counts = [0, 0, 0]
    for i in indices:
        predicted, target = set(map(tuple, predictions[i])), set(map(tuple, gold[i]))
        counts[0] += len(predicted & target)
        counts[1] += len(predicted)
        counts[2] += len(target)
    return metric(*counts)


def vote(members):
    if len(members) != 3 or len({len(x) for x in members}) != 1:
        raise ValueError("Consensus requires three aligned prediction sets")
    return [
        {span for span, count in Counter(span for member in members for span in member[i]).items() if count >= 2}
        for i in range(len(members[0]))
    ]


def consensus_rows(members):
    """Vote separately on token spans and unaligned character spans."""
    if len(members) != 3:
        raise ValueError("Consensus requires three aligned prediction sets")
    ids = [str(row["sent_id"]) for row in members[0]]
    if any([str(row["sent_id"]) for row in member] != ids for member in members[1:]):
        raise ValueError("Sentence IDs and order must match across members")
    fields = ["entities"]
    if any("unaligned_entities" in row for member in members for row in member):
        fields.append("unaligned_entities")
    result = [dict(sent_id=sent_id) for sent_id in ids]
    for field in fields:
        spans = vote([[set(map(tuple, row.get(field, []))) for row in member] for member in members])
        for row, selected in zip(result, spans):
            row[field] = sorted(selected)
    return result


def paired_bootstrap(left, right, gold, *, seed=2026, repetitions=10000):
    """Sentence bootstrap of left-minus-right F1, reported in percentage points."""
    if not len(gold) or not (len(left) == len(right) == len(gold)):
        raise ValueError("Bootstrap inputs must be nonempty and aligned")
    counts = []
    for predictions in (left, right):
        counts.append(
            np.asarray(
                [[len(set(p) & set(g)), len(set(p)), len(set(g))] for p, g in zip(predictions, gold)], dtype=np.int64
            )
        )
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(gold), len(gold))
        scores = [metric(*array[indices].sum(axis=0))["f1"] for array in counts]
        differences.append(100 * (scores[0] - scores[1]))
    low, high = np.quantile(differences, [0.025, 0.975])
    return dict(
        delta_f1=100 * (evaluate(left, gold)["f1"] - evaluate(right, gold)["f1"]),
        lower=float(low),
        upper=float(high),
        repetitions=repetitions,
        seed=seed,
    )


def member_bootstrap(left, right, gold, *, seed=2026, repetitions=10000, index_dtype="int32"):
    """Average member-specific F1 differences inside each shared sentence resample."""
    if len(left) != len(right) or not left or not gold:
        raise ValueError("Both systems need the same nonzero number of aligned members")
    if any(len(member) != len(gold) for member in left + right):
        raise ValueError("Every member must contain the same sentences as the reference")
    arrays = []
    for members in (left, right):
        arrays.append(
            np.asarray(
                [
                    [[len(set(p) & set(g)), len(set(p)), len(set(g))] for p, g in zip(member, gold)]
                    for member in members
                ],
                dtype=np.int64,
            )
        )
    rng = np.random.default_rng(seed)
    samples = []
    dtype = {"int32": np.int32, "int64": np.int64}[index_dtype]
    for begin in range(0, repetitions, 64):
        indices = rng.integers(0, len(gold), size=(min(64, repetitions - begin), len(gold)), dtype=dtype)
        scores = []
        for counts in arrays:
            total = counts[:, indices, :].sum(axis=2)
            denominator = total[..., 1] + total[..., 2]
            f1 = np.divide(
                2 * total[..., 0], denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0
            )
            scores.append(f1.mean(axis=0))
        samples.extend(100 * (scores[0] - scores[1]))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    difference = 100 * np.mean([evaluate(a, gold)["f1"] - evaluate(b, gold)["f1"] for a, b in zip(left, right)])
    return dict(
        delta_f1=float(difference),
        lower=float(lower),
        upper=float(upper),
        members=len(left),
        repetitions=repetitions,
        seed=seed,
        index_dtype=index_dtype,
    )
