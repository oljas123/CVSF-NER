"""Contextual span verifier and its packed candidate inputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, XLMRobertaConfig, XLMRobertaModel

from .io import read_json, write_json
from .metrics import entities
from .packets import make_packet, packet_indices

ENCODER = "FacebookAI/xlm-roberta-large"
ENCODER_REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"


def load_tokenizer(name=ENCODER, revision=ENCODER_REVISION):
    return AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True, fix_mistral_regex=True)


def encode_record(row, tokenizer, max_length=512, allow_truncation=False):
    full = tokenizer(row["tokens"], is_split_into_words=True, truncation=False)
    if len(full["input_ids"]) > max_length and not allow_truncation:
        raise ValueError(f"{row['sent_id']}: {len(full['input_ids'])} subwords exceed max_length={max_length}")
    encoded = tokenizer(row["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
    positions = defaultdict(list)
    for position, word in enumerate(encoded.word_ids()):
        if word is not None:
            positions[int(word)].append(position)
    counts = Counter(word for word in full.word_ids() if word is not None)
    complete = [i for i in range(len(row["tokens"])) if len(positions[i]) == counts[i] and counts[i] > 0]
    if complete != list(range(len(complete))):
        raise ValueError("Tokenizer did not preserve a contiguous word prefix")
    tags = row.get("tags_bio", ["O"] * len(row["tokens"]))
    return dict(
        sent_id=str(row["sent_id"]),
        tokens=row["tokens"],
        gold_tags=tags,
        gold=[list(span) for span in entities(tags, strict=True)],
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        word_starts=[positions[i][0] for i in complete],
        word_ends=[positions[i][-1] for i in complete],
        complete_words=len(complete),
        all_content_positions=[i for i, word in enumerate(encoded.word_ids()) if word is not None],
    )


class PoolVerifier(nn.Module):
    """Classify each proposed boundary as NONE or one of the corpus entity types."""

    def __init__(self, type_count, model_name=ENCODER, revision=ENCODER_REVISION, *, encoder=None):
        super().__init__()
        self.encoder = (
            encoder
            if encoder is not None
            else AutoModel.from_pretrained(
                model_name, revision=revision, add_pooling_layer=False, attn_implementation="eager"
            )
        )
        count = self.encoder.get_input_embeddings().num_embeddings
        self.marker_ids = (count, count + 1)
        self.encoder.resize_token_embeddings(count + 2, mean_resizing=False)
        with torch.no_grad():
            weight = self.encoder.get_input_embeddings().weight
            weight[count].copy_(weight[0])
            weight[count + 1].copy_(weight[2])
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(4 * self.encoder.config.hidden_size, type_count + 1)

    @classmethod
    def tiny(cls, type_count, vocab_size=39):
        config = XLMRobertaConfig(
            vocab_size=vocab_size,
            hidden_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=40,
            max_position_embeddings=514,
            pad_token_id=1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        config._attn_implementation = "eager"
        return cls(type_count, encoder=XLMRobertaModel(config, add_pooling_layer=False))

    def forward(self, packet, return_hidden=False):
        output = self.encoder(
            input_ids=packet["input_ids"],
            position_ids=packet["position_ids"],
            attention_mask=packet["attention_mask"],
            return_dict=True,
        )
        hidden = output.last_hidden_state[0]
        start, end = packet["starts"], packet["ends"]
        first, last = hidden[start], hidden[end]
        prefix = torch.cat([torch.zeros_like(hidden[:1]), hidden.cumsum(0)], dim=0)
        span_mean = (prefix[end + 1] - prefix[start]) / (end - start + 1).unsqueeze(1)
        sentence_mean = hidden[packet["content_positions"]].mean(0, keepdim=True).expand(len(start), -1)
        logits = self.classifier(self.dropout(torch.cat([first, last, span_mean, sentence_mean], dim=-1)))
        return (logits, hidden) if return_hidden else logits

    def save_pretrained(self, path, types):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = self.encoder.config.to_dict()
        config["vocab_size"] = self.marker_ids[0]
        write_json(path / "pool_config.json", dict(encoder=config, types=types, dropout=0.1))
        save_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in self.state_dict().items()},
            str(path / "model.safetensors"),
        )

    @classmethod
    def from_pretrained(cls, path, device="cpu"):
        path = Path(path)
        metadata = read_json(path / "pool_config.json")
        values = dict(metadata["encoder"])
        config = AutoConfig.for_model(values.pop("model_type"), **values)
        config._attn_implementation = "eager"
        encoder = AutoModel.from_config(config, add_pooling_layer=False, attn_implementation="eager")
        model = cls(len(metadata["types"]), encoder=encoder)
        model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
        return model.to(device).eval()


@torch.inference_mode()
def pool_logits(model, records, device="cpu"):
    import numpy as np

    model.eval()
    result = []
    for row in records:
        chunks = [
            model(make_packet(row, indices, model.marker_ids, device)).float().cpu().numpy()
            for indices in packet_indices(row)
        ]
        result.append(
            np.concatenate(chunks) if chunks else np.empty((0, model.classifier.out_features), dtype=np.float32)
        )
    return result


def optimizer_for(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            learning_rate = 1e-4 if name.startswith("classifier.") else 2e-5
            decay = 0.0 if name.endswith(".bias") or "LayerNorm" in name else 0.01
            groups.setdefault((learning_rate, decay), []).append(parameter)
    return torch.optim.AdamW(
        [dict(params=parameters, lr=lr, weight_decay=decay) for (lr, decay), parameters in groups.items()],
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
    )
