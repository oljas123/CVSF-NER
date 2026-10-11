"""Word-level payloads derived from complete first-subword logit vectors."""

from __future__ import annotations

from typing import Any
import numpy as np


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def token_score_payload(
    logits: np.ndarray,
    pred_id: int,
    id2label: dict[int, str],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    probs = stable_softmax(logits)
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_probs = [float(probs[int(idx)]) for idx in top_indices]
    confidence = float(probs[int(pred_id)])
    margin = top_probs[0] - top_probs[1] if len(top_probs) > 1 else confidence
    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(len(probs))) if len(probs) > 1 else 0.0
    return {
        "confidence": confidence,
        "margin": float(margin),
        "entropy": entropy,
        "top_labels": [{"label": id2label[int(idx)], "prob": float(probs[int(idx)])} for idx in top_indices],
    }
