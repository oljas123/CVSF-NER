import pytest

pytest.importorskip("torch")
from cvsf.external import align_gliner, choose_threshold, evaluate_external, filter_predictions, normalized_gliner_text


def test_unaligned_predictions_remain_false_positives():
    aligned, unaligned = align_gliner(["Ada", "Luma"], {(0, 2, "PER"): 0.9, (0, 3, "PER"): 0.8, (4, 8, "LOC"): 0.7})
    assert aligned == [[1, 2, "LOC", 0.7]]
    assert unaligned == [[0, 2, "PER", 0.9]]
    reference = [dict(sent_id="x", tokens=["Ada", "Luma"], tags_bio=["B-PER", "B-LOC"])]
    scored = [dict(sent_id="x", scored_entities=aligned, unaligned_scored_entities=unaligned)]
    result = evaluate_external(reference, filter_predictions(scored, 0.05))
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)


def test_external_threshold_tie_uses_lower_value():
    reference = [dict(sent_id="x", tags_bio=["B-PER"])]
    scored = [dict(sent_id="x", scored_entities=[[0, 1, "PER", 0.99]])]
    selected, _ = choose_threshold(reference, scored)
    assert selected["threshold"] == 0.05


def test_normalization_keeps_token_coordinates_or_fails_explicitly():
    assert normalized_gliner_text(["Ada", "Luma"], lambda text: text + ".") == "Ada Luma."
    aligned, unaligned = align_gliner(["Ada", "Luma"], {(0, 3, "PER"): 0.8, (4, 9, "LOC"): 0.7})
    assert aligned == [[0, 1, "PER", 0.8]]
    assert unaligned == [[4, 9, "LOC", 0.7]]
    with pytest.raises(ValueError, match="changed input coordinates"):
        normalized_gliner_text(["Ada", "Luma"], lambda text: " " + text)
