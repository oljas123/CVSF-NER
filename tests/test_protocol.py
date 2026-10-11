import copy

import pytest

from cvsf.cli import main
from cvsf.data import record_key, token_key, validate_folds, validate_training_data
from cvsf.io import read_json, read_jsonl, write_jsonl
from cvsf.metrics import consensus_rows
from cvsf.selection import historical_options, options, select


def record(name, tag="O"):
    return dict(sent_id=name, tokens=[name], tags_bio=[tag])


def test_text_identity_ignores_annotations_and_oof_rejects_cross_fold_duplicates():
    rows = [record("Ada", "B-PER"), record("Ada", "B-LOC"), record("Luma"), record("Riva")]
    assert token_key(rows[0]) == token_key(rows[1])
    assert record_key(rows[0]) != record_key(rows[1])
    with pytest.raises(ValueError, match="Identical training text"):
        validate_folds(rows, [dict(train=[1, 3], heldout=[0, 2]), dict(train=[0, 2], heldout=[1, 3])])
    validate_folds(rows, [dict(train=[2, 3], heldout=[0, 1]), dict(train=[0, 1], heldout=[2, 3])])


@pytest.mark.parametrize(
    "folds",
    [
        [dict(train=[2, 3], heldout=[0, 1]), dict(train=[2, 3], heldout=[0, 1])],
        [dict(train=[1, 2, 3], heldout=[0, 1]), dict(train=[0, 1], heldout=[2, 3])],
        [dict(train=[2, 3], heldout=[0, 4]), dict(train=[0, 1], heldout=[2, 3])],
    ],
)
def test_invalid_oof_coverage_fails_before_training(folds):
    with pytest.raises(ValueError):
        validate_folds([record(str(i)) for i in range(4)], folds)


def test_training_validation_checks_text_and_development_partition():
    train = [record(str(i)) for i in range(4)]
    folds = [dict(train=[2, 3], heldout=[0, 1]), dict(train=[0, 1], heldout=[2, 3])]
    with pytest.raises(ValueError, match="both train and selection"):
        validate_training_data(train, [record("0"), record("new")], folds, dict(selection=[0], comparison=[1]))
    with pytest.raises(ValueError, match="partition"):
        validate_training_data(train, [record("new1"), record("new2")], folds, dict(selection=[0], comparison=[0]))


def test_consensus_retains_unaligned_false_positives_through_cli(tmp_path):
    row = dict(sent_id="x", entities=[[0, 1, "PER"]], unaligned_entities=[[4, 6, "LOC"]])
    inputs = []
    for i in range(3):
        path = tmp_path / f"member{i}.jsonl"
        write_jsonl(path, [row])
        inputs.append(str(path))
    gold = tmp_path / "gold.jsonl"
    write_jsonl(gold, [dict(sent_id="x", tokens=["Ada", "Luma"], tags_bio=["B-PER", "O"])])
    output, metrics = tmp_path / "consensus.jsonl", tmp_path / "metrics.json"
    main(["consensus", "--predictions", *inputs, "--output", str(output)])
    assert read_jsonl(output) == [row]
    main(["evaluate", "--gold", str(gold), "--predictions", str(output), "--output", str(metrics)])
    assert {k: read_json(metrics)[k] for k in ("tp", "fp", "fn")} == dict(tp=1, fp=1, fn=0)
    assert read_json(metrics)["f1"] == 2 / 3


def test_unaligned_consensus_uses_two_distinct_members_and_separate_coordinates():
    first = dict(sent_id="x", entities=[[0, 1, "PER"]], unaligned_entities=[[0, 1, "PER"], [0, 1, "PER"]])
    second = dict(sent_id="x", entities=[[0, 1, "PER"]], unaligned_entities=[])
    result = consensus_rows([[first], [second], [second]])[0]
    assert result["entities"] == [(0, 1, "PER")] and result["unaligned_entities"] == []
    third = copy.deepcopy(second)
    third["unaligned_entities"] = [[0, 1, "PER"]]
    assert consensus_rows([[first], [second], [third]])[0]["unaligned_entities"] == [(0, 1, "PER")]
    with pytest.raises(ValueError, match="Sentence IDs"):
        consensus_rows([[first], [second], [dict(third, sent_id="wrong")]])


def test_original_draw_options_are_explicit_and_preserve_grid_predictions():
    from pathlib import Path

    config = read_json(Path(__file__).parents[1] / "configs/kyrgyz.json")
    source = [dict(bio=[], lattice=[[0, 1, "PER", 0.8]])]
    for run in config["runs"]:
        historic = historical_options(config, run, dict(dataset="kyrgyz", training_draw=2026))
        assert historic
        for system, extra in historic.items():
            assert all(option in options(system) for option in extra)
        old, _ = select(source, [{(0, 1, "PER")}], [0], ["LATTICE"], historical=historic)
        grid, _ = select(source, [{(0, 1, "PER")}], [0], ["LATTICE"])
        assert old == grid
        assert historical_options(config, run, dict(dataset="kyrgyz", training_draw=31001)) == {}
