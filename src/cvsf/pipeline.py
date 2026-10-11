"""Train a run, choose development thresholds, and predict tokenized text."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .candidates import build_memory, proposal_options
from .data import validate_training_data
from .decoding import fuse
from .features import attach_boundaries, generate_candidate_rows, group_candidates
from .io import prediction_rows, read_json, read_jsonl, sha256, tokens_only, validate_records, write_json, write_jsonl
from .lattice import fit_model, score_rows
from .metrics import entities, evaluate
from .selection import FIELDS, SYSTEMS, apply_selection, clip, historical_options, select


def candidate_rows(items, memory, prefix, *, bio_only=False):
    return generate_candidate_rows(
        items=items, memory_index=memory, external_memory_index={}, args=proposal_options(bio_only), prefix=prefix
    )


def fit(config_path, data_path, output, run="Run1", *, device="cuda", controls=True, max_steps=-1):
    from .neural import encode_record, load_tokenizer
    from .training import infer_bio, train_bio, train_pool

    config, data_path, output = read_json(config_path), Path(data_path), Path(output)
    run_config = config["runs"][run]
    seed = run_config["bio_seed"]
    schema = read_json(data_path / "label_schema.json")
    train, valid = read_jsonl(data_path / "train.jsonl"), read_jsonl(data_path / "valid.jsonl")
    folds = read_json(data_path / "folds.json")
    validate_training_data(train, valid, folds, read_json(data_path / "partition.json"))
    protocol = read_json(data_path / "protocol.json") if (data_path / "protocol.json").exists() else {}
    signature = dict(
        config=config,
        run=run,
        controls=controls,
        max_steps=max_steps,
        protocol=protocol,
        files={
            name: sha256(data_path / name)
            for name in ("train.jsonl", "valid.jsonl", "folds.json", "label_schema.json", "partition.json")
        },
    )
    if (output / "inputs.json").exists() and read_json(output / "inputs.json") != signature:
        raise ValueError("This output directory belongs to a different run; choose another --output")
    write_json(output / "inputs.json", signature)
    write_json(output / "config.json", config)
    write_json(output / "label_schema.json", schema)
    write_json(output / "run.json", dict(run=run, **run_config, controls=controls))
    if (output / "fit_complete.json").exists():
        return read_json(output / "fit_complete.json")
    tokenizer = load_tokenizer(config["encoder"], config.get("encoder_revision"))
    encoded = [
        encode_record(row, tokenizer, config["max_length"], config.get("allow_truncation", False)) for row in train
    ]
    typed, prepared = [], [None] * len(train)
    index = {str(row["sent_id"]): i for i, row in enumerate(train)}
    for number, fold in enumerate(folds):
        folder = output / "bio" / f"fold_{number}"
        training = [train[i] for i in fold["train"]]
        held = [train[i] for i in fold["heldout"]]
        if not (folder / "training.json").exists():
            train_bio(training, held, schema["labels"], config, seed, folder, device=device, max_steps=max_steps)
        predicted_path = folder / "heldout_predictions.jsonl"
        if not predicted_path.exists():
            predictions, _ = infer_bio(folder, held, config, device=device)
            write_jsonl(predicted_path, predictions)
        predictions = read_jsonl(predicted_path)
        rows, lookup = candidate_rows(predictions, build_memory(training), f"fold_{number}")
        typed.extend(rows)
        grouped = group_candidates(rows, lookup)
        for key, item in lookup.items():
            i = index[str(item["sent_id"])]
            if prepared[i] is not None:
                raise ValueError("A training sentence occurs in more than one held-out fold")
            prepared[i] = attach_boundaries(encoded[i], grouped[key], schema["types"])
    if any(row is None for row in prepared) or {row["label"] for row in typed} != {0, 1}:
        raise ValueError("OOF candidates must cover every training sentence and contain both outcome classes")
    write_jsonl(output / "oof_typed_candidates.jsonl", typed)
    write_jsonl(output / "pool_training_candidates.jsonl", prepared)
    if not (output / "lattice.joblib").exists():
        joblib.dump(fit_model(typed, "extratrees", seed), output / "lattice.joblib")
    memory = build_memory(train)
    write_json(output / "memory.json", {surface: dict(counts) for surface, counts in memory.items()})
    for name, training_seed in [("primary", seed)] + ([("partner", run_config["partner_seed"])] if controls else []):
        folder = output / "bio" / name
        if not (folder / "training.json").exists():
            train_bio(train, valid, schema["labels"], config, training_seed, folder, device=device, max_steps=max_steps)
    if not (output / "pool" / "training.json").exists():
        train_pool(prepared, schema["types"], config, run_config["pool_seed"], output / "pool", device=device)
    receipt = dict(
        run=run,
        training_sentences=len(train),
        typed_candidates=len(typed),
        candidate_boundaries=sum(len(row["spans"]) for row in prepared),
        controls=controls,
    )
    write_json(output / "fit_complete.json", receipt)
    return receipt


def items_from_probabilities(records, probabilities, labels):
    items = []
    for record, values in zip(records, probabilities):
        tokens = record["tokens"][: len(values)]
        words, tags = [], []
        for i, probability in enumerate(values):
            choice = int(probability.argmax())
            ranked = np.argsort(probability, kind="quicksort")[::-1][:3]
            nonzero = probability[probability > 0]
            tags.append(labels[choice])
            words.append(
                dict(
                    index=i,
                    token=tokens[i],
                    gold_tag="O",
                    pred_tag=labels[choice],
                    confidence=float(probability[choice]),
                    margin=float(probability[ranked[0]] - probability[ranked[1]]),
                    entropy=float(-np.sum(nonzero * np.log(nonzero)) / np.log(len(probability))),
                    top_labels=[dict(label=labels[int(k)], prob=float(probability[k])) for k in ranked],
                )
            )
        items.append(
            dict(
                sent_id=str(record["sent_id"]),
                tokens=tokens,
                gold_tags=["O"] * len(tokens),
                pred_tags=tags,
                token_scores=words,
            )
        )
    return items


def probabilities(logits):
    result = []
    for array in logits:
        z = np.asarray(array, dtype=np.float64)
        exponent = np.exp(z - z.max(axis=1, keepdims=True))
        result.append(exponent / exponent.sum(axis=1, keepdims=True))
    return result


def inventory(items, memory, lattice=None):
    raw, lookup = candidate_rows(items, {}, "evaluation", bio_only=True)
    bio = group_candidates(raw, lookup)
    if lattice is not None:
        rows, lookup = candidate_rows(items, memory, "evaluation")
        grouped = group_candidates(score_rows(rows, lattice), lookup)
    else:
        rows, grouped = [], {key: [] for key in lookup}
    output = []
    for key, item in lookup.items():
        output.append(
            dict(
                sent_id=str(item["sent_id"]),
                bio=[
                    dict(
                        s=int(r["start"]),
                        e=int(r["end"]),
                        t=str(r["pred_type"]),
                        scores={field: float(r[field]) for field in FIELDS},
                    )
                    for r in bio[key]
                ],
                lattice=[
                    [int(r["start"]), int(r["end"]), str(r["pred_type"]), float(r["reranker_probability"])]
                    for r in grouped[key]
                ],
            )
        )
    return output, rows, lookup


def score(model_path, records, *, device="cuda", ablation_threshold=None):
    from collections import Counter
    from .neural import PoolVerifier, encode_record, load_tokenizer, pool_logits
    from .training import infer_bio, release

    model_path = Path(model_path)
    config, schema = read_json(model_path / "config.json"), read_json(model_path / "label_schema.json")
    records = tokens_only(records)
    validate_records(records, labeled=False)
    memory = {surface: Counter(counts) for surface, counts in read_json(model_path / "memory.json").items()}
    lattice = joblib.load(model_path / "lattice.joblib")
    _, primary_logits = infer_bio(model_path / "bio" / "primary", records, config, device=device)
    primary_probabilities = probabilities(primary_logits)
    primary, rows, lookup = inventory(
        items_from_probabilities(records, primary_probabilities, schema["labels"]), memory, lattice
    )
    grouped = group_candidates(rows, lookup)
    by_id = {str(item["sent_id"]): key for key, item in lookup.items()}
    tokenizer = load_tokenizer(str(model_path / "bio" / "primary"), revision=None)
    prepared = [
        attach_boundaries(
            encode_record(row, tokenizer, config["max_length"], config.get("allow_truncation", False)),
            grouped[by_id[str(row["sent_id"])]],
            schema["types"],
        )
        for row in records
    ]
    if read_json(model_path / "run.json")["controls"]:
        _, partner_logits = infer_bio(model_path / "bio" / "partner", records, config, device=device)
        partner_probabilities = probabilities(partner_logits)
        partner, _, _ = inventory(items_from_probabilities(records, partner_probabilities, schema["labels"]), memory)
        paired = [(a + b) / 2 for a, b in zip(primary_probabilities, partner_probabilities)]
        average, _, _ = inventory(items_from_probabilities(records, paired, schema["labels"]), memory, lattice)
        for a, b, combined in zip(primary, partner, average):
            a["pair_bio"], a["pair_lattice"] = combined["bio"], combined["lattice"]
            a["pair_entity"] = {}
            for field in FIELDS:
                left = {(r["s"], r["e"], r["t"]): clip(r["scores"][field]) for r in a["bio"]}
                right = {(r["s"], r["e"], r["t"]): clip(r["scores"][field]) for r in b["bio"]}
                a["pair_entity"][field] = [
                    [*span, (left.get(span, 0.0) + right.get(span, 0.0)) / 2]
                    for span in sorted(left.keys() | right.keys())
                ]
    model = PoolVerifier.from_pretrained(model_path / "pool", device)
    logits = pool_logits(model, prepared, device)
    for row, candidate, values in zip(primary, prepared, logits):
        row.update(fuse(candidate["spans"], row["lattice"], values, schema["types"]))
        if ablation_threshold is not None:
            from .ablations import ablate

            row["ablations"] = ablate(candidate["spans"], row, values, schema["types"], ablation_threshold)
    release(model)
    return primary


def select_development(model_path, data_path, *, device="cuda"):
    model_path, data_path = Path(model_path), Path(data_path)
    records = read_jsonl(data_path / "valid.jsonl")
    validate_records(records)
    partition = read_json(data_path / "partition.json")
    expected = read_json(model_path / "inputs.json")["files"]
    if any(sha256(data_path / name) != expected[name] for name in ("valid.jsonl", "partition.json")):
        raise ValueError("Development files differ from this run's training configuration")
    source = score(model_path, records, device=device)
    gold = [set(entities(row["tags_bio"], strict=True)) for row in records]
    systems = SYSTEMS if read_json(model_path / "run.json")["controls"] else SYSTEMS[:5]
    historical = historical_options(
        read_json(model_path / "config.json"),
        read_json(model_path / "run.json")["run"],
        read_json(model_path / "inputs.json").get("protocol", {}),
    )
    decisions, curves = select(source, gold, partition["selection"], systems, historical=historical)
    predictions = apply_selection(source, decisions)
    write_json(model_path / "selection.json", decisions)
    write_json(model_path / "selection_curves.json", curves)
    write_json(
        model_path / "development_metrics.json",
        {
            split: {system: evaluate(prediction, gold, partition[split]) for system, prediction in predictions.items()}
            for split in ("selection", "comparison")
        },
    )
    write_jsonl(model_path / "development_scores.jsonl", source)
    return decisions


def predict_file(model_path, input_path, output, *, device="cuda"):
    model_path, output = Path(model_path), Path(output)
    decisions = read_json(model_path / "selection.json")
    records = tokens_only(read_jsonl(input_path))
    source = score(model_path, records, device=device)
    predictions = apply_selection(source, decisions)
    for system, spans in predictions.items():
        write_jsonl(output / f"{system}.jsonl", prediction_rows(records, spans))
    write_json(
        output / "prediction_manifest.json",
        dict(
            sentences=len(records),
            systems=list(predictions),
            input_sha256=sha256(input_path),
            selection_sha256=sha256(model_path / "selection.json"),
        ),
    )
    return predictions


def ablate_file(model_path, input_path, output, *, device="cuda"):
    model_path, output = Path(model_path), Path(output)
    threshold = read_json(model_path / "selection.json")["FUSION"]["option"]["threshold"]
    records = tokens_only(read_jsonl(input_path))
    source = score(model_path, records, device=device, ablation_threshold=threshold)
    variants = list(source[0]["ablations"]) if source else []
    for name in variants:
        write_jsonl(output / f"{name}.jsonl", prediction_rows(records, [row["ablations"][name] for row in source]))
    write_json(
        output / "ablation_manifest.json",
        dict(
            variants=variants,
            fixed_threshold=threshold,
            selection_sha256=sha256(model_path / "selection.json"),
            input_sha256=sha256(input_path),
        ),
    )
    return dict(variants=variants, fixed_threshold=threshold)
