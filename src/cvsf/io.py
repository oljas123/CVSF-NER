"""Small UTF-8 JSON utilities shared by the command-line tools."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens_only(rows):
    return [dict(sent_id=str(row["sent_id"]), tokens=list(row["tokens"])) for row in rows]


def validate_records(rows, *, labeled=True):
    from .metrics import entities

    ids = []
    for row in rows:
        ids.append(str(row["sent_id"]))
        tokens = row["tokens"]
        if not tokens or not all(isinstance(token, str) and token for token in tokens):
            raise ValueError(f"Empty or invalid tokens in {ids[-1]}")
        if labeled:
            if len(row["tags_bio"]) != len(tokens):
                raise ValueError(f"Token/tag length mismatch in {ids[-1]}")
            entities(row["tags_bio"], strict=True)
    if len(ids) != len(set(ids)):
        raise ValueError("Sentence IDs must be unique within a split")


def prediction_rows(records, predictions):
    if len(records) != len(predictions):
        raise ValueError("Prediction and input lengths differ")
    return [dict(sent_id=str(row["sent_id"]), entities=sorted(spans)) for row, spans in zip(records, predictions)]
