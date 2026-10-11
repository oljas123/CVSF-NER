"""Typed candidates from BIO predictions, local boundaries and training memory."""

from __future__ import annotations

import re
from argparse import Namespace
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .metrics import extract_entities

SPACE_RE = re.compile(r"\s+")


def normalize_surface(text: str | None) -> str:
    if not text:
        return ""
    return SPACE_RE.sub(" ", str(text).casefold().strip())


@dataclass(frozen=True)
class Candidate:
    sent_key: str
    sent_id: str
    start: int
    end: int
    ent_type: str
    source: str
    score: float | None = None

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (self.sent_key, self.start, self.end, self.ent_type)

    @property
    def span_key(self) -> tuple[str, int, int]:
        return (self.sent_key, self.start, self.end)


def split_bio(label: str | None) -> tuple[str, str | None]:
    if not label:
        return "O", None
    label = str(label)
    if label == "O" or "-" not in label:
        return label, None
    prefix, ent_type = label.split("-", 1)
    return prefix, ent_type


def label_prob(token_score: dict[str, Any], label: str) -> float:
    for item in token_score.get("top_labels", []):
        if item.get("label") == label:
            return float(item.get("prob") or 0.0)
    if token_score.get("pred_tag") == label:
        return float(token_score.get("confidence") or 0.0)
    return 0.0


def top_labels(token_score: dict[str, Any], *, top_k: int) -> list[tuple[str, float]]:
    labels = []
    for item in token_score.get("top_labels", [])[:top_k]:
        labels.append((str(item.get("label")), float(item.get("prob") or 0.0)))
    return labels


def token_entity_probability(token_score: dict[str, Any], *, top_k: int) -> float:
    o_prob = label_prob(token_score, "O")
    if o_prob > 0:
        return max(0.0, 1.0 - o_prob)
    entity_probs = [prob for label, prob in top_labels(token_score, top_k=top_k) if split_bio(label)[0] in {"B", "I"}]
    return max(entity_probs) if entity_probs else 0.0


def candidate_score(item: dict[str, Any], start: int, end: int, ent_type: str) -> float | None:
    token_scores = item.get("token_scores", [])
    if len(token_scores) != len(item.get("tokens", [])):
        return None
    probs = []
    for idx in range(start, end):
        label = f"B-{ent_type}" if idx == start else f"I-{ent_type}"
        prob = label_prob(token_scores[idx], label)
        if prob <= 0.0:
            prob = float(token_scores[idx].get("confidence") or 0.0)
        probs.append(prob)
    return sum(probs) / len(probs) if probs else None


def bio_candidates(item: dict[str, Any], sent_key: str) -> list[Candidate]:
    rows = []
    for entity in extract_entities(item.get("pred_tags", []), item.get("tokens", [])):
        start = int(entity["start"])
        end = int(entity["end"])
        ent_type = str(entity["type"])
        rows.append(
            Candidate(
                sent_key=sent_key,
                sent_id=str(item.get("sent_id")),
                start=start,
                end=end,
                ent_type=ent_type,
                source="bio_argmax",
                score=candidate_score(item, start, end, ent_type),
            )
        )
    return rows


def boundary_candidates(item: dict[str, Any], sent_key: str) -> list[Candidate]:
    rows = []
    n_tokens = len(item.get("tokens", []))
    seen: set[tuple[int, int, str]] = set()
    offsets = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1)]
    for entity in extract_entities(item.get("pred_tags", []), item.get("tokens", [])):
        start = int(entity["start"])
        end = int(entity["end"])
        ent_type = str(entity["type"])
        for left_delta, right_delta in offsets:
            new_start = start + left_delta
            new_end = end + right_delta
            if new_start < 0 or new_end > n_tokens or new_start >= new_end:
                continue
            key = (new_start, new_end, ent_type)
            if key in seen or (new_start == start and new_end == end):
                continue
            seen.add(key)
            rows.append(
                Candidate(
                    sent_key=sent_key,
                    sent_id=str(item.get("sent_id")),
                    start=new_start,
                    end=new_end,
                    ent_type=ent_type,
                    source="boundary_perturbation",
                    score=candidate_score(item, new_start, new_end, ent_type),
                )
            )
    return rows


def top_label_lattice_candidates(
    item: dict[str, Any],
    sent_key: str,
    *,
    top_k: int,
    min_start_prob: float,
    min_continue_prob: float,
    min_mean_prob: float,
    max_span_len: int,
) -> list[Candidate]:
    token_scores = item.get("token_scores", [])
    rows = []
    for start, token_score in enumerate(token_scores):
        for label, start_prob in top_labels(token_score, top_k=top_k):
            prefix, ent_type = split_bio(label)
            if prefix != "B" or ent_type is None or start_prob < min_start_prob:
                continue
            probs = [start_prob]
            max_end = min(len(token_scores), start + max_span_len)
            for end in range(start + 1, max_end + 1):
                mean_prob = sum(probs) / len(probs)
                if mean_prob >= min_mean_prob:
                    rows.append(
                        Candidate(
                            sent_key=sent_key,
                            sent_id=str(item.get("sent_id")),
                            start=start,
                            end=end,
                            ent_type=ent_type,
                            source="top_label_lattice",
                            score=mean_prob,
                        )
                    )
                if end >= max_end:
                    break
                next_prob = label_prob(token_scores[end], f"I-{ent_type}")
                if next_prob < min_continue_prob:
                    break
                probs.append(next_prob)
    return rows


def memory_candidates(
    item: dict[str, Any],
    sent_key: str,
    memory_index: dict[str, Counter[str]],
    *,
    max_span_len: int,
    max_types_per_span: int,
    source_name: str = "memory_span",
) -> list[Candidate]:
    if not memory_index:
        return []
    tokens = item.get("tokens", [])
    rows = []
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + max_span_len) + 1):
            surface = normalize_surface(" ".join(tokens[start:end]))
            type_counts = memory_index.get(surface)
            if not type_counts:
                continue
            for ent_type, count in type_counts.most_common(max_types_per_span):
                rows.append(
                    Candidate(
                        sent_key=sent_key,
                        sent_id=str(item.get("sent_id")),
                        start=start,
                        end=end,
                        ent_type=ent_type,
                        source=source_name,
                        score=float(count),
                    )
                )
    return rows


def ngram_lattice_candidates(
    item: dict[str, Any],
    sent_key: str,
    *,
    top_k: int,
    max_span_len: int,
    min_entity_prob: float,
    max_types_per_span: int,
) -> list[Candidate]:
    token_scores = item.get("token_scores", [])
    tokens = item.get("tokens", [])
    if len(token_scores) != len(tokens):
        return []
    bio_spans = [
        (int(entity["start"]), int(entity["end"]), str(entity["type"]))
        for entity in extract_entities(item.get("pred_tags", []), tokens)
    ]
    rows = []
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + max_span_len) + 1):
            overlaps_bio = any(not (end <= left or start >= right) for left, right, _ in bio_spans)
            entity_prob = sum(token_entity_probability(token_scores[idx], top_k=top_k) for idx in range(start, end)) / (
                end - start
            )
            if not overlaps_bio and entity_prob < min_entity_prob:
                continue
            type_scores: Counter[str] = Counter()
            for label, prob in top_labels(token_scores[start], top_k=top_k):
                prefix, ent_type = split_bio(label)
                if prefix == "B" and ent_type:
                    type_scores[ent_type] += prob
            for idx in range(start + 1, end):
                for label, prob in top_labels(token_scores[idx], top_k=top_k):
                    prefix, ent_type = split_bio(label)
                    if prefix == "I" and ent_type:
                        type_scores[ent_type] += prob
            for left, right, ent_type in bio_spans:
                if not (end <= left or start >= right):
                    type_scores[ent_type] += 0.5
            if not type_scores:
                continue
            for ent_type, type_score in type_scores.most_common(max_types_per_span):
                rows.append(
                    Candidate(
                        sent_key=sent_key,
                        sent_id=str(item.get("sent_id")),
                        start=start,
                        end=end,
                        ent_type=ent_type,
                        source="ngram_lattice",
                        score=float(type_score),
                    )
                )
    return rows


def dedupe(candidates: Iterable[Candidate]) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    merged: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = candidate.key
        if key not in merged:
            merged[key] = {
                "sent_key": candidate.sent_key,
                "sent_id": candidate.sent_id,
                "start": candidate.start,
                "end": candidate.end,
                "type": candidate.ent_type,
                "sources": {candidate.source},
                "score": candidate.score,
            }
            continue
        merged[key]["sources"].add(candidate.source)
        old_score = merged[key].get("score")
        if old_score is None or (candidate.score is not None and candidate.score > old_score):
            merged[key]["score"] = candidate.score
    return merged


def sentence_key(idx: int, item: dict[str, Any]) -> str:
    return f"{idx}::{item.get('sent_id')}"


def proposal_options(bio_only=False):
    """The candidate settings used for every language in the paper."""
    return Namespace(
        disable_top_label=bio_only,
        disable_boundary=bio_only,
        disable_memory=bio_only,
        disable_external_memory=True,
        include_ngram=not bio_only,
        boundary_gate="none",
        boundary_gate_min_mean_prob=0.05,
        boundary_gate_min_token_prob=0.01,
        top_k=3,
        max_span_len=8,
        max_types_per_span=3,
        min_start_prob=1e-6,
        min_continue_prob=1e-6,
        min_mean_prob=1e-6,
        ngram_min_entity_prob=0.01,
        external_memory_require_existing_candidate=True,
    )


def build_memory(records):
    """Count case-folded entity surfaces in the supplied training records."""
    index = defaultdict(Counter)
    for record in records:
        for entity in extract_entities(record["tags_bio"], record["tokens"]):
            surface = normalize_surface(entity["text"])
            if surface:
                index[surface][entity["type"].upper()] += 1
    return index
