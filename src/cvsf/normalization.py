"""Corpus tag normalization and span-preserving Doccano tokenization."""

from __future__ import annotations

import re
from typing import Any, Iterable


def normalize_tag(tag: object) -> str:
    text = str(tag).strip()
    if not text or text.lower() == "nan" or text == "O":
        return "O"
    text = text.replace("_", "-")
    if "-" not in text:
        return text.upper()
    prefix, ent_type = text.split("-", 1)
    ent_type = ent_type.strip().upper().replace(" ", "_")
    return f"{prefix.strip().upper()}-{ent_type}"


def split_tag(tag: str) -> tuple[str, str | None]:
    tag = normalize_tag(tag)
    if tag == "O":
        return "O", None
    if "-" not in tag:
        return tag, None
    prefix, ent_type = tag.split("-", 1)
    return prefix, ent_type


def bioes_to_bio(tags: Iterable[str], *, repair: bool = True) -> list[str]:
    converted: list[str] = []
    for tag in tags:
        prefix, ent_type = split_tag(tag)
        if prefix == "O" or ent_type is None:
            converted.append("O")
        elif prefix == "S":
            converted.append(f"B-{ent_type}")
        elif prefix == "E":
            converted.append(f"I-{ent_type}")
        else:
            converted.append(f"{prefix}-{ent_type}")
    return repair_iob2(converted) if repair else converted


def repair_iob2(tags: Iterable[str]) -> list[str]:
    repaired: list[str] = []
    active_type: str | None = None
    for tag in tags:
        prefix, ent_type = split_tag(tag)
        if prefix == "O" or ent_type is None:
            repaired.append("O")
            active_type = None
        elif prefix == "B":
            repaired.append(f"B-{ent_type}")
            active_type = ent_type
        elif prefix == "I" and active_type == ent_type:
            repaired.append(f"I-{ent_type}")
        elif prefix == "I":
            repaired.append(f"B-{ent_type}")
            active_type = ent_type
        else:
            repaired.append(f"B-{ent_type}")
            active_type = ent_type
    return repaired


def map_bio_tags(tags: Iterable[str], mapping: dict[str, str]) -> list[str]:
    mapped: list[str] = []
    normalized_mapping = {key.upper(): value.upper() for key, value in mapping.items()}
    for tag in tags:
        prefix, entity_type = split_tag(tag)
        mapped_type = normalized_mapping.get(entity_type or "")
        if prefix not in {"B", "I"} or mapped_type is None:
            mapped.append("O")
        else:
            mapped.append(f"{prefix}-{mapped_type}")
    return repair_iob2(mapped)


def span_tokenize_doccano(
    text: str,
    raw_labels: list[Any],
) -> tuple[list[str], list[str], list[list[int]], list[list[Any]], int]:
    spans: list[tuple[int, int, str]] = []
    for raw_label in raw_labels:
        if not isinstance(raw_label, (list, tuple)) or len(raw_label) != 3:
            raise ValueError(f"invalid Doccano label: {raw_label!r}")
        start, end, entity_type = raw_label
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"non-integer Doccano span: {raw_label!r}")
        if not 0 <= start < end <= len(text):
            raise ValueError(f"Doccano span outside text: {raw_label!r} for text length {len(text)}")
        spans.append((start, end, str(entity_type).strip().upper()))
    spans.sort(key=lambda item: (item[0], item[1], item[2]))
    for previous, current in zip(spans, spans[1:]):
        if previous[1] > current[0]:
            raise ValueError(f"overlapping Doccano spans: {previous!r}, {current!r}")

    boundaries = {offset for start, end, _ in spans for offset in (start, end)}
    tokens: list[str] = []
    offsets: list[list[int]] = []
    boundary_splits = 0
    for match in re.finditer(r"\S+", text):
        internal = sorted(offset for offset in boundaries if match.start() < offset < match.end())
        cuts = [match.start(), *internal, match.end()]
        boundary_splits += len(internal)
        for start, end in zip(cuts, cuts[1:]):
            token = text[start:end]
            if token:
                tokens.append(token)
                offsets.append([start, end])

    if not tokens:
        raise ValueError("empty sentence after span-aware tokenization")

    tags = ["O"] * len(tokens)
    normalized_spans: list[list[Any]] = []
    for entity_start, entity_end, entity_type in spans:
        token_indices = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_start < entity_end and token_end > entity_start
        ]
        if not token_indices:
            raise ValueError(
                f"Doccano span contains no non-whitespace token: {(entity_start, entity_end, entity_type)!r}"
            )
        if any(offsets[index][0] < entity_start or offsets[index][1] > entity_end for index in token_indices):
            raise ValueError(
                f"partial token/span overlap after boundary splitting: {(entity_start, entity_end, entity_type)!r}"
            )
        if any(tags[index] != "O" for index in token_indices):
            raise ValueError(
                f"multiple Doccano spans assigned to one token: {(entity_start, entity_end, entity_type)!r}"
            )
        for position, token_index in enumerate(token_indices):
            prefix = "B" if position == 0 else "I"
            tags[token_index] = f"{prefix}-{entity_type}"
        normalized_spans.append([entity_start, entity_end, entity_type])

    return tokens, tags, offsets, normalized_spans, boundary_splits
