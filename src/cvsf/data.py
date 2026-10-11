"""Retrieve original corpora and reconstruct the recorded experiment splits."""

from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

from .io import read_json, sha256, validate_records, write_json, write_jsonl
from .normalization import bioes_to_bio, map_bio_tags, span_tokenize_doccano


def record_key(row):
    payload = json.dumps([row["tokens"], row["tags_bio"]], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token_key(row):
    payload = json.dumps(row["tokens"], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checked_indices(values, size, name):
    if not values or any(type(index) is not int or not 0 <= index < size for index in values):
        raise ValueError(f"{name}: indices must be nonempty integers within the split")
    if len(set(values)) != len(values):
        raise ValueError(f"{name}: repeated indices")
    return set(values)


def validate_folds(records, folds):
    """Check complete OOF coverage and keep equal token sequences in one fold."""
    if len(folds) < 2:
        raise ValueError("OOF training needs at least two folds")
    all_indices, seen, text_folds = set(range(len(records))), [], {}
    for number, fold in enumerate(folds):
        train = checked_indices(fold["train"], len(records), f"Fold {number} train")
        held = checked_indices(fold["heldout"], len(records), f"Fold {number} heldout")
        if train & held or train | held != all_indices:
            raise ValueError(f"Fold {number}: train and heldout must partition the training records")
        seen.extend(held)
        for index in held:
            key = token_key(records[index])
            if key in text_folds and text_folds[key] != number:
                raise ValueError("Identical training text occurs in different OOF heldout folds")
            text_folds[key] = number
    if sorted(seen) != list(range(len(records))):
        raise ValueError("Each training record must occur in exactly one heldout fold")


def validate_partition(records, partition):
    selection = checked_indices(partition["selection"], len(records), "Dev-S")
    comparison = checked_indices(partition["comparison"], len(records), "Dev-C")
    if selection & comparison or selection | comparison != set(range(len(records))):
        raise ValueError("Dev-S and Dev-C must partition the development records")


def validate_split_texts(splits):
    keys = {name: {token_key(row) for row in records} for name, records in splits.items()}
    names = list(keys)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if keys[left] & keys[right]:
                raise ValueError(f"Identical text occurs in both {left} and {right}")


def validate_training_data(train, valid, folds, partition):
    validate_records(train)
    validate_records(valid)
    validate_folds(train, folds)
    validate_partition(valid, partition)
    validate_split_texts(
        dict(
            train=train,
            selection=[valid[i] for i in partition["selection"]],
            comparison=[valid[i] for i in partition["comparison"]],
        )
    )


def read_conll(path, *, comment_prefix=None, mapping=None):
    records, tokens, tags = [], [], []

    def flush():
        if tokens:
            bio = bioes_to_bio(tags)
            if mapping:
                bio = map_bio_tags(bio, mapping)
            records.append(dict(tokens=list(tokens), tags_bio=bio))
            tokens.clear()
            tags.clear()

    with Path(path).open(encoding="utf-8-sig", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if not line.strip() or (comment_prefix and line.startswith(comment_prefix)):
                flush()
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) >= 2 and parts[0].strip():
                tokens.append(parts[0].strip())
                tags.append(parts[-1].strip())
    flush()
    return records


def load_source(source, directory):
    result = []
    for file in source["files"]:
        path = Path(directory) / file["name"]
        if sha256(path) != file["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        if source["format"] == "conll":
            result.extend(
                read_conll(path, comment_prefix=source.get("comment_prefix"), mapping=source.get("tag_mapping"))
            )
        elif source["format"] == "doccano_gzip":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        tokens, tags, *_ = span_tokenize_doccano(row["text"], row["labels"])
                        result.append(dict(tokens=tokens, tags_bio=tags))
        else:
            raise ValueError(f"Unknown corpus format: {source['format']}")
    return result


def download_source(source, directory, *, offline=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for file in source["files"]:
        path = directory / file["name"]
        if path.exists():
            if sha256(path) != file["sha256"]:
                raise ValueError(f"Cached file has a different checksum: {path}")
            continue
        if offline:
            raise FileNotFoundError(f"Place the original source file at {path}, or omit --offline")
        request = urllib.request.Request(file["url"], headers={"User-Agent": "CVSF-NER/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != file["sha256"]:
            raise ValueError(f"Downloaded file differs from the recorded corpus: {file['url']}")
        path.write_bytes(content)


def prepare(registry_path, manifest_path, cache, output, *, offline=False):
    registry, manifest = read_json(registry_path), read_json(manifest_path)
    source = registry[manifest["dataset"]]
    folder = Path(cache) / manifest["dataset"]
    download_source(source, folder, offline=offline)
    available = {record_key(row): row for row in load_source(source, folder)}
    output = Path(output)
    identity = dict(manifest_sha256=sha256(manifest_path), dataset=manifest["dataset"])
    if (output / "preparation.json").exists() and read_json(output / "preparation.json")["identity"] != identity:
        raise ValueError("The output folder contains another data split; choose a different --output")
    splits = {}
    for split, members in manifest["splits"].items():
        missing = [member["record_sha256"] for member in members if member["record_sha256"] not in available]
        if missing:
            raise ValueError(f"{split}: {len(missing)} records do not match the pinned source and normalization")
        records = [dict(sent_id=member["sent_id"], **available[member["record_sha256"]]) for member in members]
        validate_records(records)
        if any(member.get("token_sha256", token_key(row)) != token_key(row) for member, row in zip(members, records)):
            raise ValueError(f"{split}: token sequence checksum differs from the manifest")
        splits[split] = records
    validate_training_data(splits["train"], splits["valid"], manifest["folds"], manifest["partition"])
    validate_split_texts(splits)
    for split, records in splits.items():
        write_jsonl(output / f"{split}.jsonl", records)
    write_json(output / "label_schema.json", manifest["schema"])
    write_json(output / "folds.json", manifest["folds"])
    write_json(output / "partition.json", manifest["partition"])
    write_json(output / "protocol.json", dict(dataset=manifest["dataset"], training_draw=manifest["training_draw"]))
    receipt = dict(
        identity=identity,
        counts={name: len(rows) for name, rows in splits.items()},
        files={path.name: sha256(path) for path in sorted(output.glob("*.jsonl"))},
    )
    write_json(output / "preparation.json", receipt)
    return receipt
