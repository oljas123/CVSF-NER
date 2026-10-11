"""Development operating points for the fusion and internal controls."""

from __future__ import annotations

from .decoding import decode
from .metrics import evaluate

FIELDS = (
    "candidate_score",
    "min_expected_prob",
    "mean_token_confidence",
    "min_token_confidence",
    "mean_margin",
    "min_margin",
)
GRID = tuple(i / 20 for i in range(1, 20))
SYSTEMS = (
    "BIO_RAW",
    "FILTER",
    "LATTICE",
    "POOL",
    "FUSION",
    "PAIR_ENTITY_FILTER",
    "PAIR_TOKEN_FILTER",
    "PAIR_TOKEN_LATTICE",
)


def clip(probability):
    return max(1e-6, min(1 - 1e-6, float(probability)))


def options(system, historical=None):
    historical = list((historical or {}).get(system, []))
    if system == "BIO_RAW":
        return [dict(kind="raw", threshold=0.0)]
    if system in ("FILTER", "PAIR_TOKEN_FILTER", "PAIR_ENTITY_FILTER"):
        first = dict(kind="single" if system == "PAIR_ENTITY_FILTER" else "raw", threshold=0.0)
        return (
            [first]
            + historical
            + [dict(kind="scalar", field=field, threshold=threshold) for field in FIELDS for threshold in GRID]
        )
    return historical + [dict(kind="scored", threshold=threshold) for threshold in GRID]


def historical_options(config, run, protocol):
    """Restore the eligible options for the original Kyrgyz training draw."""
    settings = config.get("selection", {})
    if (
        protocol.get("dataset") == config.get("dataset") == "kyrgyz"
        and protocol.get("training_draw") == settings.get("historical_draw") == 2026
    ):
        return settings["historical_options"][run]
    return {}


def predict(source, system, option, indices=None, selected_filter=None):
    indices = list(range(len(source))) if indices is None else list(indices)
    predictions = []
    for i in indices:
        row = source[i]
        kind = option["kind"]
        if kind == "single":
            predictions.append(selected_filter[i])
            continue
        bio = row["pair_bio"] if system == "PAIR_TOKEN_FILTER" else row["bio"]
        if kind == "raw":
            predictions.append({(int(r["s"]), int(r["e"]), str(r["t"])) for r in bio})
            continue
        if kind == "scalar":
            spans = (
                row["pair_entity"][option["field"]]
                if system == "PAIR_ENTITY_FILTER"
                else [[r["s"], r["e"], r["t"], clip(r["scores"][option["field"]])] for r in bio]
            )
        else:
            spans = row[{"LATTICE": "lattice", "PAIR_TOKEN_LATTICE": "pair_lattice"}.get(system, system)]
        predictions.append(decode(spans, option["threshold"]))
    return predictions


def select(source, gold, indices, systems=SYSTEMS, *, historical=None):
    decisions, curves, filtered = {}, [], None
    targets = [gold[i] for i in indices]
    for system in systems:
        choices = []
        for option in options(system, historical):
            prediction = predict(source, system, option, indices, filtered)
            choices.append(dict(option=option, selection=evaluate(prediction, targets)))
        best = max(
            choices,
            key=lambda row: (
                row["selection"]["f1"],
                row["selection"]["precision"],
                -row["selection"]["pred"],
                row["option"]["threshold"],
            ),
        )
        decisions[system] = best
        curves.extend(dict(system=system, **choice) for choice in choices)
        if system == "FILTER":
            filtered = predict(source, system, best["option"])
    return decisions, curves


def apply_selection(source, decisions):
    result, filtered = {}, None
    for system in SYSTEMS:
        if system not in decisions:
            continue
        result[system] = predict(source, system, decisions[system]["option"], selected_filter=filtered)
        if system == "FILTER":
            filtered = result[system]
    return result
