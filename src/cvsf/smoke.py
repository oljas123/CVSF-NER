"""A complete CPU run with synthetic sentences and a locally initialized encoder."""

from __future__ import annotations

from pathlib import Path

from .io import read_jsonl, write_json, write_jsonl


def run(output):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel, set_seed
    from .metrics import entities, evaluate
    from .pipeline import fit, predict_file, select_development

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Smoke test output must be a new or empty directory")
    torch.set_num_threads(2)
    set_seed(7)
    words = ["<s>", "<pad>", "</s>", "<unk>", "Ada", "Bo", "Cy", "Dee", "Eon", "Fen", "visited", "Riva", "Luma", "."]
    words.extend(f"id{i}" for i in range(40))
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.post_processor = TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=64,
    )
    encoder_path = output / "tiny_encoder"
    tokenizer.save_pretrained(encoder_path)
    model_config = XLMRobertaConfig(
        vocab_size=len(words),
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=40,
        max_position_embeddings=128,
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
    )
    XLMRobertaModel(model_config, add_pooling_layer=False).save_pretrained(encoder_path)
    records = [
        dict(
            sent_id=f"synthetic-{i}",
            tokens=[words[4 + i % 6], "visited", "Riva" if i % 2 else "Luma", ".", f"id{i}"],
            tags_bio=["B-PER", "O", "B-LOC", "O", "O"],
        )
        for i in range(28)
    ]
    data = output / "data"
    write_jsonl(data / "train.jsonl", records[:20])
    write_jsonl(data / "valid.jsonl", records[20:26])
    write_jsonl(data / "test.jsonl", records[26:])
    write_json(
        data / "folds.json",
        [
            dict(train=[i for i in range(20) if i % 5 != f], heldout=[i for i in range(20) if i % 5 == f])
            for f in range(5)
        ],
    )
    write_json(data / "partition.json", dict(selection=[0, 1, 2], comparison=[3, 4, 5]))
    write_json(data / "label_schema.json", dict(types=["LOC", "PER"], labels=["O", "B-LOC", "B-PER", "I-LOC", "I-PER"]))
    config = dict(
        encoder=str(encoder_path),
        encoder_revision=None,
        max_length=64,
        bio=dict(epochs=1, learning_rate=3e-5, bf16=False, inference_bf16=False),
        pool=dict(epochs=1),
        runs=dict(Run1=dict(bio_seed=7, partner_seed=2007, pool_seed=7, external_seed=7)),
    )
    config_path = output / "config.json"
    write_json(config_path, config)
    fitted = output / "run"
    fit(config_path, data, fitted, device="cpu", max_steps=2)
    select_development(fitted, data, device="cpu")
    predictions = predict_file(fitted, data / "test.jsonl", output / "predictions", device="cpu")
    gold = [set(entities(row["tags_bio"])) for row in read_jsonl(data / "test.jsonl")]
    scores = {system: evaluate(values, gold) for system, values in predictions.items()}
    write_json(output / "smoke_result.json", dict(passed=True, synthetic=True, device="cpu", systems=scores))
    return dict(passed=True, systems=len(scores), output=str(output))
