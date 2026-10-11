"""BIO training, word-level inference, and verifier optimization."""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .io import write_json
from .neural import PoolVerifier, encode_record, load_tokenizer, optimizer_for
from .packets import make_packet, packet_indices
from .token_scores import token_score_payload


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def release(model=None):
    if model is not None:
        model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TokenDataset(Dataset):
    def __init__(self, records, tokenizer, labels, max_length, allow_truncation=False):
        label_ids = {label: i for i, label in enumerate(labels)}
        self.features = []
        for row in records:
            encode_record(row, tokenizer, max_length, allow_truncation)
            encoded = tokenizer(row["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
            target, previous = [], None
            for word in encoded.word_ids():
                target.append(label_ids[row["tags_bio"][word]] if word is not None and word != previous else -100)
                previous = word
            self.features.append(dict(encoded, labels=target))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def train_bio(train, valid, labels, config, seed, destination, *, device="cuda", max_steps=-1):
    configure()
    set_seed(seed)
    destination = Path(destination)
    tokenizer = load_tokenizer(config["encoder"], config.get("encoder_revision"))
    model = AutoModelForTokenClassification.from_pretrained(
        config["encoder"],
        revision=config.get("encoder_revision"),
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
    )
    recipe = config["bio"]
    dataset_args = (tokenizer, labels, config["max_length"], config.get("allow_truncation", False))
    training = TokenDataset(train, *dataset_args)
    validation = TokenDataset(valid, *dataset_args)
    arguments = TrainingArguments(
        output_dir=str(destination / "trainer"),
        do_train=True,
        do_eval=True,
        num_train_epochs=recipe["epochs"],
        max_steps=max_steps,
        learning_rate=recipe["learning_rate"],
        weight_decay=0.01,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        disable_tqdm=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
        fp16=False,
        bf16=recipe.get("bf16", False) and device != "cpu",
        optim="adamw_torch",
        remove_unused_columns=True,
        use_cpu=device == "cpu",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=training,
        eval_dataset=validation,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    start = time.perf_counter()
    trainer.train()
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(destination)
    write_json(
        destination / "training.json",
        dict(
            seed=seed,
            steps=trainer.state.global_step,
            epochs=trainer.state.epoch,
            seconds=time.perf_counter() - start,
            history=trainer.state.log_history,
            checkpoint="final_epoch",
            arguments=arguments.to_dict(),
        ),
    )
    del trainer
    release(model)


@torch.inference_mode()
def infer_bio(model_path, records, config, *, device="cpu", batch_size=4):
    configure()
    tokenizer = load_tokenizer(str(model_path), revision=None)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device).eval()
    labels = {int(key): value for key, value in model.config.id2label.items()}
    predictions, all_logits = [], []
    for begin in range(0, len(records), batch_size):
        batch = records[begin : begin + batch_size]
        for row in batch:
            encode_record(row, tokenizer, config["max_length"], config.get("allow_truncation", False))
        encoded = tokenizer(
            [row["tokens"] for row in batch],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt",
        )
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=config["bio"].get("inference_bf16", False) and device != "cpu",
        ):
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits.float().cpu().numpy()
        for i, row in enumerate(batch):
            positions, seen = [], set()
            for position, word in enumerate(encoded.word_ids(i)):
                if word is not None and word not in seen:
                    positions.append(position)
                    seen.add(word)
            z = np.asarray(logits[i, positions], dtype=np.float32)
            all_logits.append(z)
            predictions.append(prediction_from_logits(row, z, labels))
    release(model)
    return predictions, all_logits


def prediction_from_logits(row, logits, labels):
    tokens = row["tokens"][: len(logits)]
    tags = row.get("tags_bio", ["O"] * len(row["tokens"]))[: len(logits)]
    predicted = logits.argmax(axis=1)
    return dict(
        sent_id=str(row["sent_id"]),
        tokens=tokens,
        gold_tags=tags,
        pred_tags=[labels[int(label)] for label in predicted],
        token_scores=[
            dict(
                token_score_payload(z, int(label), labels),
                index=i,
                token=tokens[i],
                gold_tag=tags[i],
                pred_tag=labels[int(label)],
            )
            for i, (z, label) in enumerate(zip(logits, predicted))
        ],
    )


def pool_update(model, optimizer, scheduler, rows, indices, *, device="cpu"):
    denominator = sum(len(rows[int(i)]["spans"]) for i in indices)
    if not denominator:
        raise ValueError("An effective training batch has no proposed spans")
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for index in indices:
        row = rows[int(index)]
        for chunk in packet_indices(row):
            packet = make_packet(row, chunk, model.marker_ids, device)
            with torch.autocast(
                device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"
            ):
                loss = F.cross_entropy(model(packet).float(), packet["targets"], reduction="sum") / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError("Verifier loss is not finite")
            loss.backward()
            total_loss += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return total_loss


def train_pool(rows, types, config, seed, destination, *, device="cuda"):
    configure()
    set_seed(seed)
    model = PoolVerifier(len(types), config["encoder"], config.get("encoder_revision")).to(device)
    optimizer = optimizer_for(model)
    epochs = config["pool"]["epochs"]
    total_steps = epochs * math.ceil(len(rows) / 8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / total_steps))
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(rows))
        losses = [
            pool_update(model, optimizer, scheduler, rows, order[begin : begin + 8], device=device)
            for begin in range(0, len(rows), 8)
        ]
        log = dict(epoch=epoch + 1, steps=scheduler.last_epoch, loss=float(np.mean(losses)))
        history.append(log)
        print(log, flush=True)
    model.save_pretrained(destination, types)
    write_json(
        Path(destination) / "training.json",
        dict(
            seed=seed,
            history=history,
            steps=scheduler.last_epoch,
            seconds=time.perf_counter() - start,
            parameter_dtype="float32",
            training_autocast="bfloat16" if device.startswith("cuda") else "float32",
        ),
    )
    release(model)
