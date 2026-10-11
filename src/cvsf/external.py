"""SpanMarker and GLiNER2.5 adapters for the paper's external comparisons."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .io import read_json, read_jsonl, tokens_only, validate_records, write_json, write_jsonl
from .metrics import entities, metric
from .selection import GRID


def offsets(tokens):
    starts, ends, position = [], [], 0
    for token in tokens:
        starts.append(position)
        position += len(token)
        ends.append(position)
        position += 1
    return " ".join(tokens), starts, ends


def gliner_example(row, names):
    text, starts, ends = offsets(row["tokens"])
    output = {name: [] for name in names.values()}
    for start, end, typ in entities(row["tags_bio"], strict=True):
        left, right = starts[start], ends[end - 1]
        output[names[typ]].append(dict(text=text[left:right], start=left, end=right))
    return dict(input=text, output=dict(entities=output))


def filter_predictions(predictions, threshold):
    return [
        dict(
            sent_id=row["sent_id"],
            entities=[p[:3] for p in row["scored_entities"] if p[3] >= threshold],
            unaligned_entities=[p[:3] for p in row.get("unaligned_scored_entities", []) if p[3] >= threshold],
        )
        for row in predictions
    ]


def evaluate_external(records, predictions):
    if len(records) != len(predictions):
        raise ValueError("Prediction and reference lengths differ")
    tp = predicted = gold_count = 0
    for row, prediction in zip(records, predictions):
        if str(row["sent_id"]) != str(prediction["sent_id"]):
            raise ValueError("Prediction sentence order differs from the reference")
        gold, spans = set(entities(row["tags_bio"], strict=True)), set(map(tuple, prediction["entities"]))
        tp += len(gold & spans)
        gold_count += len(gold)
        predicted += len(spans) + len(prediction.get("unaligned_entities", []))
    return metric(tp, predicted, gold_count)


def choose_threshold(records, predictions):
    curve = [
        dict(threshold=threshold, **evaluate_external(records, filter_predictions(predictions, threshold)))
        for threshold in GRID
    ]
    return max(curve, key=lambda row: (row["f1"], -row["threshold"])), curve


def load_model(arm, config, schema, *, checkpoint=None, device="cuda"):
    from huggingface_hub import snapshot_download

    if arm == "spanmarker":
        from span_marker import SpanMarkerConfig, SpanMarkerModel

        if checkpoint is None:
            encoder = snapshot_download(
                config["encoder"],
                revision=config["encoder_revision"],
                allow_patterns=["*.json", "*.safetensors", "*.model"],
            )
            model = SpanMarkerModel.from_pretrained(
                encoder,
                labels=schema["labels"],
                model_max_length=512,
                marker_max_length=64,
                entity_max_length=config["external"]["entity_max_length"],
                attn_implementation="eager",
            )
        else:
            saved_config = SpanMarkerConfig.from_pretrained(str(checkpoint))
            saved_config.encoder["_attn_implementation"] = "eager"
            model = SpanMarkerModel.from_pretrained(str(checkpoint), config=saved_config, attn_implementation="eager")
        if model.encoder.config._attn_implementation != "eager":
            raise ValueError("SpanMarker must use eager attention for checkpoint-compatible predictions")
    elif arm == "gliner":
        from gliner2 import AutoExtractor

        source = (
            str(checkpoint)
            if checkpoint
            else snapshot_download(
                config["external"]["gliner_model"],
                revision=config["external"]["gliner_revision"],
                allow_patterns=["*.json", "*.safetensors", "*.model", "encoder_config/*"],
            )
        )
        model = AutoExtractor.from_pretrained(source, map_location="cpu")
        model.processor.sampling_config.synthetic_entity_label_prob = 0.0
    else:
        raise ValueError(f"Unknown external model: {arm}")
    return model.float().to(device)


def align_gliner(tokens, candidates):
    """Flatten character spans, keeping non-token-aligned predictions for scoring."""
    _, starts, ends = offsets(tokens)
    start_to_word = {position: i for i, position in enumerate(starts)}
    end_to_word = {position: i + 1 for i, position in enumerate(ends)}
    scored, unaligned, occupied = [], [], set()
    for (start, end, typ), confidence in sorted(candidates.items(), key=lambda item: (-item[1], *item[0])):
        if occupied.intersection(range(start, end)):
            continue
        occupied.update(range(start, end))
        if start not in start_to_word or end not in end_to_word:
            unaligned.append([start, end, typ, confidence])
        else:
            scored.append([start_to_word[start], end_to_word[end], typ, confidence])
    return sorted(scored), sorted(unaligned)


def normalized_gliner_text(tokens, normalize):
    """Allow the pinned processor's terminal period without shifting word offsets."""
    original = " ".join(tokens)
    normalized = normalize(original)
    if normalized not in (original, original + "."):
        raise ValueError("GLiNER normalization changed input coordinates; a character-offset mapping is required")
    return normalized


@torch.inference_mode()
def predict(model, arm, config, records, *, batch_size=2):
    records = tokens_only(records)
    was_training, saved_forward = model.training, model.forward
    if hasattr(model, "_original_forward"):
        model.forward = model._original_forward
    model.eval()
    result = []
    try:
        with torch.autocast(device_type=next(model.parameters()).device.type, enabled=False):
            if arm == "spanmarker":
                outputs = model.predict(
                    [row["tokens"] for row in records], batch_size=batch_size, show_progress_bar=False
                )
                for row, output in zip(records, outputs):
                    result.append(
                        dict(
                            sent_id=row["sent_id"],
                            scored_entities=sorted(
                                [
                                    int(entity["word_start_index"]),
                                    int(entity["word_end_index"]),
                                    entity["label"],
                                    float(entity["score"]),
                                ]
                                for entity in output
                            ),
                        )
                    )
            else:
                names = config["external"]["gliner_label_names"]
                reverse = {name: typ for typ, name in names.items()}
                texts = [normalized_gliner_text(row["tokens"], model.processor._normalize_text) for row in records]
                outputs = model.batch_extract_entities(
                    texts,
                    list(names.values()),
                    batch_size=batch_size,
                    threshold=0.05,
                    include_confidence=True,
                    include_spans=True,
                    max_len=4096,
                    overlap_policy="allow",
                )
                for row, text, output in zip(records, texts, outputs):
                    candidates = {}
                    for label, values in output["entities"].items():
                        for entity in values:
                            start, end = int(entity["start"]), int(entity["end"])
                            if text[start:end] != entity["text"]:
                                raise ValueError("GLiNER returned inconsistent character offsets")
                            key = (start, end, reverse[label])
                            candidates[key] = max(candidates.get(key, 0.0), float(entity["confidence"]))
                    scored, unaligned = align_gliner(row["tokens"], candidates)
                    result.append(
                        dict(sent_id=row["sent_id"], scored_entities=scored, unaligned_scored_entities=unaligned)
                    )
    finally:
        model.forward = saved_forward
        model.train(was_training)
    if len(result) != len(records):
        raise ValueError("External model returned a different number of sentences")
    return result


class Selection:
    def __init__(self, arm, config, records, seed, output):
        self.arm, self.config, self.records, self.seed = arm, config, records, seed
        self.output, self.best, self.history = Path(output), None, []

    def evaluate(self, model, epoch, step):
        scored = predict(model, self.arm, self.config, self.records)
        if self.arm == "gliner":
            selected, curve = choose_threshold(self.records, scored)
        else:
            selected = dict(threshold=0.0, **evaluate_external(self.records, filter_predictions(scored, 0.0)))
            curve = [selected]
        event = dict(epoch=epoch, step=step, **selected)
        self.history.append(event)
        if self.best is None or event["f1"] > self.best["f1"]:
            self.best = event
            model.save_pretrained(str(self.output / "best"))
            write_json(self.output / "selection.json", dict(**event, curve=curve))
            write_jsonl(self.output / "selection_predictions.jsonl", filter_predictions(scored, selected["threshold"]))
        write_json(self.output / "selection_history.json", self.history)
        print(event, flush=True)
        return event


def train_span(model, manager, records, schema, max_steps=-1):
    from datasets import Dataset
    from span_marker import Trainer
    from transformers import TrainingArguments

    lookup = {label: i for i, label in enumerate(schema["labels"])}

    def dataset(rows):
        return Dataset.from_dict(
            dict(
                tokens=[row["tokens"] for row in rows],
                ner_tags=[[lookup[tag] for tag in row["tags_bio"]] for row in rows],
            )
        )

    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    class ExactTrainer(Trainer):
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            event = manager.evaluate(self.model, float(self.state.epoch or 0), int(self.state.global_step))
            values = {"eval_f1": event["f1"]}
            self.log(values)
            return values

    cpu = next(model.parameters()).device.type == "cpu"
    args = TrainingArguments(
        output_dir=str(manager.output / "trainer"),
        num_train_epochs=manager.config["external"]["epochs"],
        max_steps=max_steps,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        bf16=not cpu,
        fp16=False,
        tf32=False,
        gradient_checkpointing=False,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=20,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=not cpu,
        seed=manager.seed,
        data_seed=manager.seed,
        optim="adamw_torch",
        disable_tqdm=True,
        save_safetensors=True,
        remove_unused_columns=False,
        use_cpu=cpu,
    )
    trainer = ExactTrainer(
        model=model, args=args, train_dataset=dataset(records), eval_dataset=dataset(manager.records)
    )
    trainer.model_accepts_loss_kwargs = False
    result = trainer.train()
    if manager.best is None:
        manager.evaluate(model, float(trainer.state.epoch), trainer.state.global_step)
    return dict(global_steps=trainer.state.global_step, epochs=trainer.state.epoch, metrics=result.metrics)


def train_gliner(model, manager, records, max_steps=-1):
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    class ExactTrainer(ExtractorTrainer):
        def _evaluate(self, eval_dataset):
            event = manager.evaluate(self.model, int(self.epoch) + 1, int(self.global_step))
            self.best_metric = manager.best["f1"]
            metrics = dict(eval_f1=event["f1"], epoch=int(self.epoch) + 1, step=int(self.global_step))
            self.eval_metrics_history.append(metrics)
            return metrics

        def _save_checkpoint(self, name):
            # Selection saves the best weights after each epoch's evaluation.
            write_json(
                manager.output / "progress.json",
                dict(step=int(self.global_step), epoch=int(self.epoch) + 1, event=name),
            )

    config = TrainingConfig(
        output_dir=str(manager.output / "trainer"),
        experiment_name=manager.output.name,
        num_epochs=manager.config["external"]["epochs"],
        max_steps=max_steps,
        batch_size=1,
        eval_batch_size=2,
        gradient_accumulation_steps=8,
        encoder_lr=1e-5,
        task_lr=5e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        scheduler_type="linear",
        bf16=next(model.parameters()).device.type == "cuda",
        fp16=False,
        eval_strategy="epoch",
        save_best=False,
        metric_for_best="eval_f1",
        greater_is_better=True,
        early_stopping=False,
        logging_steps=20,
        num_workers=0,
        report_to_wandb=False,
        seed=manager.seed,
        max_len=4096,
        strict_training=True,
        ignore_nonfinite_losses=False,
        skip_step_errors=False,
        allow_invalid_samples=False,
        on_capacity_exceeded="raise",
        gradient_checkpointing=False,
        allow_tf32=False,
        float32_matmul_precision="highest",
        fused_optimizer=False,
    )
    names = manager.config["external"]["gliner_label_names"]
    return ExactTrainer(model, config).train(
        train_data=[gliner_example(row, names) for row in records],
        eval_data=[gliner_example(row, names) for row in manager.records],
    )


def train_external(config_path, data_path, output, arm, run="Run1", *, device="cuda", max_steps=-1):
    from transformers import set_seed
    from .training import configure, release

    output, data_path = Path(output), Path(data_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("External training needs a new output directory")
    config, schema = read_json(config_path), read_json(data_path / "label_schema.json")
    train, valid = read_jsonl(data_path / "train.jsonl"), read_jsonl(data_path / "valid.jsonl")
    from .data import validate_training_data

    partition = read_json(data_path / "partition.json")
    validate_training_data(train, valid, read_json(data_path / "folds.json"), partition)
    selection = [valid[i] for i in partition["selection"]]
    seed = config["runs"][run]["external_seed"]
    configure()
    set_seed(seed)
    model = load_model(arm, config, schema, device=device)
    write_json(output / "config.json", config)
    write_json(output / "label_schema.json", schema)
    write_json(output / "run.json", dict(arm=arm, run=run, seed=seed))
    manager = Selection(arm, config, selection, seed, output)
    started = time.perf_counter()
    result = (
        train_span(model, manager, train, schema, max_steps)
        if arm == "spanmarker"
        else train_gliner(model, manager, train, max_steps)
    )
    write_json(output / "training.json", dict(seconds=time.perf_counter() - started, best=manager.best, result=result))
    release(model)
    return manager.best


def predict_external(model_path, input_path, output, *, device="cuda"):
    from .training import configure, release

    model_path = Path(model_path)
    config, schema = read_json(model_path / "config.json"), read_json(model_path / "label_schema.json")
    arm = read_json(model_path / "run.json")["arm"]
    configure()
    model = load_model(arm, config, schema, checkpoint=model_path / "best", device=device)
    records = tokens_only(read_jsonl(input_path))
    validate_records(records, labeled=False)
    scored = predict(model, arm, config, records)
    predictions = filter_predictions(scored, read_json(model_path / "selection.json")["threshold"])
    write_jsonl(output, predictions)
    release(model)
    return predictions
