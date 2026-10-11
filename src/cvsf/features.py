"""The 25 numeric and two categorical inputs to the lattice classifier."""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from .candidates import (
    bio_candidates,
    boundary_candidates,
    dedupe,
    label_prob,
    memory_candidates,
    ngram_lattice_candidates,
    sentence_key,
    token_entity_probability,
    top_label_lattice_candidates,
)
from .metrics import extract_entities

BASE_NUMERIC_FEATURES = [
    "span_len",
    "relative_start",
    "relative_end",
    "candidate_score",
    "source_count",
    "source_bio",
    "source_top_label",
    "source_boundary",
    "source_memory",
    "source_external_memory",
    "source_ngram",
    "mean_expected_prob",
    "min_expected_prob",
    "start_expected_prob",
    "end_expected_prob",
    "mean_token_confidence",
    "min_token_confidence",
    "mean_margin",
    "min_margin",
    "mean_entropy",
    "max_entropy",
    "mean_entity_prob",
    "max_entity_prob",
    "left_context_entity_prob",
    "right_context_entity_prob",
]


NUMERIC_FEATURES = list(BASE_NUMERIC_FEATURES)


CATEGORICAL_FEATURES = ["pred_type", "source_signature"]


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_min(values: list[float]) -> float:
    return min(values) if values else 0.0


def safe_max(values: list[float]) -> float:
    return max(values) if values else 0.0


def boundary_label_support(item: dict[str, Any], candidate: Any) -> tuple[float, float]:
    token_scores = item.get("token_scores", [])
    tokens = item.get("tokens", [])
    if len(token_scores) != len(tokens):
        return 0.0, 0.0
    start = int(candidate.start)
    end = int(candidate.end)
    ent_type = str(candidate.ent_type)
    probs = [
        label_prob(token_scores[idx], f"B-{ent_type}" if idx == start else f"I-{ent_type}") for idx in range(start, end)
    ]
    return safe_mean(probs), safe_min(probs)


def keep_boundary_candidate(item: dict[str, Any], candidate: Any, args: argparse.Namespace) -> bool:
    if args.boundary_gate == "none":
        return True
    mean_prob, min_prob = boundary_label_support(item, candidate)
    return mean_prob >= args.boundary_gate_min_mean_prob and min_prob >= args.boundary_gate_min_token_prob


def candidate_feature_row(
    *,
    item: dict[str, Any],
    sent_key: str,
    candidate: dict[str, Any],
    label: int,
) -> dict[str, Any]:
    tokens = item.get("tokens", [])
    token_scores = item.get("token_scores", [])
    start = int(candidate["start"])
    end = int(candidate["end"])
    pred_type = str(candidate["type"])
    sources = set(candidate["sources"])
    expected = []
    confidences = []
    margins = []
    entropies = []
    entity_probs = []
    if len(token_scores) == len(tokens):
        for idx in range(start, end):
            label_name = f"B-{pred_type}" if idx == start else f"I-{pred_type}"
            expected.append(label_prob(token_scores[idx], label_name))
            confidences.append(float(token_scores[idx].get("confidence") or 0.0))
            margins.append(float(token_scores[idx].get("margin") or 0.0))
            entropies.append(float(token_scores[idx].get("entropy") or 0.0))
            entity_probs.append(token_entity_probability(token_scores[idx], top_k=3))
    sent_len = len(tokens) or 1
    source_signature = "+".join(sorted(sources))
    return {
        "sent_key": sent_key,
        "sent_id": str(item.get("sent_id")),
        "dataset": item.get("dataset"),
        "split": item.get("split"),
        "language": item.get("language"),
        "style": item.get("style"),
        "start": start,
        "end": end,
        "text": " ".join(tokens[start:end]),
        "pred_type": pred_type,
        "label": int(label),
        "source_signature": source_signature,
        "span_len": end - start,
        "relative_start": start / sent_len,
        "relative_end": end / sent_len,
        "candidate_score": float(candidate.get("score") or 0.0),
        "source_count": len(sources),
        "source_bio": int("bio_argmax" in sources),
        "source_top_label": int("top_label_lattice" in sources),
        "source_boundary": int("boundary_perturbation" in sources),
        "source_memory": int("memory_span" in sources),
        "source_external_memory": int("external_memory_span" in sources),
        "source_ngram": int("ngram_lattice" in sources),
        "mean_expected_prob": safe_mean(expected),
        "min_expected_prob": safe_min(expected),
        "start_expected_prob": expected[0] if expected else 0.0,
        "end_expected_prob": expected[-1] if expected else 0.0,
        "mean_token_confidence": safe_mean(confidences),
        "min_token_confidence": safe_min(confidences),
        "mean_margin": safe_mean(margins),
        "min_margin": safe_min(margins),
        "mean_entropy": safe_mean(entropies),
        "max_entropy": safe_max(entropies),
        "mean_entity_prob": safe_mean(entity_probs),
        "max_entity_prob": safe_max(entity_probs),
        "left_context_entity_prob": (
            token_entity_probability(token_scores[start - 1], top_k=3)
            if len(token_scores) == len(tokens) and start > 0
            else 0.0
        ),
        "right_context_entity_prob": (
            token_entity_probability(token_scores[end], top_k=3)
            if len(token_scores) == len(tokens) and end < len(tokens)
            else 0.0
        ),
    }


def generate_candidate_rows(
    *,
    items: list[dict[str, Any]],
    memory_index: dict[str, Counter[str]],
    external_memory_index: dict[str, Counter[str]] | None = None,
    args: argparse.Namespace,
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    item_by_key: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(items):
        base_key = sentence_key(idx, item)
        sent_key = f"{prefix}::{base_key}"
        item_by_key[sent_key] = item
        candidates = []
        candidates.extend(bio_candidates(item, sent_key))
        if not args.disable_boundary:
            boundary_rows = boundary_candidates(item, sent_key)
            if args.boundary_gate != "none":
                boundary_rows = [cand for cand in boundary_rows if keep_boundary_candidate(item, cand, args)]
            candidates.extend(boundary_rows)
        if not args.disable_top_label:
            candidates.extend(
                top_label_lattice_candidates(
                    item,
                    sent_key,
                    top_k=args.top_k,
                    min_start_prob=args.min_start_prob,
                    min_continue_prob=args.min_continue_prob,
                    min_mean_prob=args.min_mean_prob,
                    max_span_len=args.max_span_len,
                )
            )
        if not args.disable_memory:
            candidates.extend(
                memory_candidates(
                    item,
                    sent_key,
                    memory_index,
                    max_span_len=args.max_span_len,
                    max_types_per_span=args.max_types_per_span,
                )
            )
        if external_memory_index and not getattr(args, "disable_external_memory", False):
            external_candidates = memory_candidates(
                item,
                sent_key,
                external_memory_index,
                max_span_len=args.max_span_len,
                max_types_per_span=args.max_types_per_span,
                source_name="external_memory_span",
            )
            if getattr(args, "external_memory_require_existing_candidate", False):
                existing = {(cand.start, cand.end, cand.ent_type) for cand in candidates}
                external_candidates = [
                    cand for cand in external_candidates if (cand.start, cand.end, cand.ent_type) in existing
                ]
            candidates.extend(external_candidates)
        if args.include_ngram:
            candidates.extend(
                ngram_lattice_candidates(
                    item,
                    sent_key,
                    top_k=args.top_k,
                    max_span_len=args.max_span_len,
                    min_entity_prob=args.ngram_min_entity_prob,
                    max_types_per_span=args.max_types_per_span,
                )
            )
        merged = dedupe(candidates)
        gold = {
            (sent_key, int(entity["start"]), int(entity["end"]), str(entity["type"]))
            for entity in extract_entities(item.get("gold_tags", []), item.get("tokens", []))
        }
        for candidate in merged.values():
            key = (sent_key, int(candidate["start"]), int(candidate["end"]), str(candidate["type"]))
            rows.append(
                candidate_feature_row(
                    item=item,
                    sent_key=sent_key,
                    candidate=candidate,
                    label=int(key in gold),
                )
            )
    return rows, item_by_key


def group_candidates(rows, item_by_key):
    grouped = {key: [] for key in item_by_key}
    for row in rows:
        grouped[row["sent_key"]].append(row)
    return grouped


def attach_boundaries(encoded, candidates, types):
    """Gold is consulted only AFTER the proposed boundary inventory is fixed."""
    bounds = sorted({(int(r["start"]), int(r["end"])) for r in candidates})
    assert all(0 <= s < e <= len(encoded["tokens"]) for s, e in bounds)
    representable = [(s, e) for s, e in bounds if e <= encoded["complete_words"]]
    gold = {(int(s), int(e)): str(t) for s, e, t in encoded["gold"]}
    assert len(gold) == len(encoded["gold"]), "Multiple types on an identical gold boundary"
    result = {
        **encoded,
        "spans": [list(x) for x in representable],
        "targets": [1 + types.index(gold[b]) if b in gold else 0 for b in representable],
        "excluded_bounds": [list(b) for b in bounds if b not in set(representable)],
    }
    return result
