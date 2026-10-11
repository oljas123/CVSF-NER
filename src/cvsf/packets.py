"""Packed span inputs with directed marker attention."""

from __future__ import annotations

import torch


def make_packet(row, indices, marker_ids, device="cpu"):
    """Directed mask uses 0 for allowed attention and FP32 minimum for blocked."""
    spans = [row["spans"][i] for i in indices]
    text = row["input_ids"]
    length, count = len(text), len(spans)
    assert count and count <= 64
    a = [row["word_starts"][s] for s, e in spans]
    b = [row["word_ends"][e - 1] for s, e in spans]
    assert all(0 < s <= e < length - 1 for s, e in zip(a, b))
    ids = text + [v for _ in spans for v in marker_ids]
    positions = list(range(2, length + 2))
    positions += [v for s, e in zip(a, b) for v in (s + 2, e + 2)]
    total = length + 2 * count
    visible = torch.zeros((total, total), dtype=torch.bool)
    visible[:, :length] = True
    for i in range(count):
        start = length + 2 * i
        visible[start : start + 2, start : start + 2] = True
    additive = torch.zeros((total, total), dtype=torch.float32)
    additive.masked_fill_(~visible, torch.finfo(torch.float32).min)
    packet = {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "position_ids": torch.tensor([positions], dtype=torch.long),
        "attention_mask": additive[None, None],
        "starts": torch.tensor(a, dtype=torch.long),
        "ends": torch.tensor(b, dtype=torch.long),
        "marker_starts": torch.arange(length, total, 2),
        "marker_ends": torch.arange(length + 1, total, 2),
        "content_positions": torch.tensor(row["all_content_positions"], dtype=torch.long),
        "targets": torch.tensor([row["targets"][i] for i in indices], dtype=torch.long),
    }
    return {k: v.to(device) for k, v in packet.items()}


def packet_indices(row, pack=64):
    for begin in range(0, len(row["spans"]), pack):
        yield list(range(begin, min(begin + pack, len(row["spans"]))))
