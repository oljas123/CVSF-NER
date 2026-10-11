import copy

from cvsf.candidates import build_memory, proposal_options
from cvsf.features import generate_candidate_rows
from cvsf.normalization import map_bio_tags, span_tokenize_doccano


def item():
    return dict(
        sent_id="test",
        tokens=["Ada", "visited", "Luma"],
        pred_tags=["B-PER", "O", "O"],
        gold_tags=["B-PER", "O", "B-LOC"],
        token_scores=[
            dict(
                pred_tag=pred,
                confidence=0.7,
                margin=0.5,
                entropy=0.4,
                top_labels=[dict(label=pred, prob=0.7), dict(label="B-LOC", prob=0.2), dict(label="I-PER", prob=0.1)],
            )
            for pred in ["B-PER", "O", "O"]
        ],
    )


def test_candidates_do_not_depend_on_prediction_gold():
    observed = item()
    memory = build_memory([dict(tokens=["Luma"], tags_bio=["B-LOC"])])
    rows, _ = generate_candidate_rows(items=[observed], memory_index=memory, args=proposal_options(), prefix="x")
    hidden = copy.deepcopy(observed)
    hidden["gold_tags"] = ["O"] * 3
    blind, _ = generate_candidate_rows(items=[hidden], memory_index=memory, args=proposal_options(), prefix="x")
    assert [{k: v for k, v in row.items() if k != "label"} for row in rows] == [
        {k: v for k, v in row.items() if k != "label"} for row in blind
    ]
    assert any((row["start"], row["end"], row["pred_type"]) == (2, 3, "LOC") and row["source_memory"] for row in rows)


def test_memory_uses_only_supplied_rows_and_normalizes_surface():
    memory = build_memory([dict(tokens=["ADA"], tags_bio=["B-PER"])])
    assert dict(memory["ada"]) == {"PER": 1}
    assert "luma" not in memory


def test_doccano_splits_at_character_span_boundaries():
    tokens, tags, offsets, _, count = span_tokenize_doccano("preAda met Luma.", [[3, 6, "PER"], [11, 15, "LOC"]])
    assert tokens == ["pre", "Ada", "met", "Luma", "."]
    assert tags == ["O", "B-PER", "O", "B-LOC", "O"]
    assert offsets[1] == [3, 6] and count == 2


def test_uyghur_mapping_preserves_adjacent_entity_boundaries():
    assert map_bio_tags(["B-NHCS", "I-NHCS", "B-NHVI", "B-NT"], {"NHCS": "PER", "NHVI": "PER", "NT": "TIME"}) == [
        "B-PER",
        "I-PER",
        "B-PER",
        "B-TIME",
    ]


def test_source_local_type_limits_preserve_the_union():
    from collections import Counter

    observed = dict(
        sent_id="x",
        tokens=["Ada"],
        gold_tags=["O"],
        pred_tags=["B-PER"],
        token_scores=[
            dict(
                confidence=0.5,
                margin=0.2,
                entropy=0.8,
                pred_tag="B-PER",
                top_labels=[
                    dict(label="B-PER", prob=0.5),
                    dict(label="B-LOC", prob=0.3),
                    dict(label="B-ORG", prob=0.2),
                ],
            )
        ],
    )
    rows, _ = generate_candidate_rows(
        items=[observed], memory_index={"ada": Counter(MISC=3, TIME=2, EVENT=1)}, args=proposal_options(), prefix="x"
    )
    assert {row["pred_type"] for row in rows if (row["start"], row["end"]) == (0, 1)} == {
        "PER",
        "LOC",
        "ORG",
        "MISC",
        "TIME",
        "EVENT",
    }
