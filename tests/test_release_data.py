from pathlib import Path
import csv
import itertools
import shutil

import pytest

from cvsf.io import read_json, sha256, write_json
from cvsf.results import verify

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_folds_and_development_partitions_are_disjoint():
    for path in (ROOT / "manifests").glob("*.json"):
        manifest = read_json(path)
        train = manifest["splits"]["train"]
        seen = []
        for fold in manifest["folds"]:
            assert not set(fold["train"]) & set(fold["heldout"])
            assert set(fold["train"]) | set(fold["heldout"]) == set(range(len(train)))
            seen.extend(fold["heldout"])
            assert not (
                {train[i]["token_sha256"] for i in fold["train"]} & {train[i]["token_sha256"] for i in fold["heldout"]}
            )
        assert sorted(seen) == list(range(len(train)))
        split = manifest["partition"]
        assert not set(split["selection"]) & set(split["comparison"])
        assert sorted(split["selection"] + split["comparison"]) == list(range(len(manifest["splits"]["valid"])))
        parts = {name: {row["token_sha256"] for row in rows} for name, rows in manifest["splits"].items()}
        assert all(not left & right for left, right in itertools.combinations(parts.values(), 2))
        valid = manifest["splits"]["valid"]
        assert not (
            {valid[i]["token_sha256"] for i in split["selection"]}
            & {valid[i]["token_sha256"] for i in split["comparison"]}
        )


def test_paper_counts_reproduce_all_reported_run_means():
    result = verify(ROOT / "results/paper")
    assert result["individual_rows"] == 108
    assert result["group_rows"] == 36
    assert result["external_rows"] == result["interval_rows"] == 24


@pytest.mark.parametrize("duplicate", [False, True])
def test_result_validator_rejects_missing_or_duplicate_runs_even_with_updated_hashes(tmp_path, duplicate):
    folder = tmp_path / "paper"
    shutil.copytree(ROOT / "results/paper", folder)
    path = folder / "main_results_individual.csv"
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    lines = lines + [lines[-1]] if duplicate else lines[:-1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksums = read_json(folder / "checksums.json")
    checksums[path.name] = sha256(path)
    write_json(folder / "checksums.json", checksums)
    with pytest.raises(ValueError, match="coverage"):
        verify(folder)


def test_recorded_internal_operating_points_are_eligible_for_their_draw():
    from cvsf.selection import historical_options, options

    with (ROOT / "results/paper/internal_selection.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 168
    for row in rows:
        config = read_json(ROOT / "configs" / (row["language"] + ".json"))
        protocol = dict(
            dataset=row["language"],
            training_draw=int(row["training_draw"]) if row["training_draw"].isdigit() else row["training_draw"],
        )
        historical = historical_options(config, "Run" + row["run_id"], protocol)
        expected = dict(kind=row["kind"], threshold=float(row["threshold"]))
        if row["field"]:
            expected["field"] = row["field"]
        assert expected in options(row["system"], historical)
        if row["training_draw"] != "2026":
            assert not historical


def test_training_subset_scores_cannot_drift_from_paired_differences(tmp_path):
    folder = tmp_path / "paper"
    shutil.copytree(ROOT / "results/paper", folder)
    path = folder / "training_draw_scores.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    row = next(row for row in rows if row["draw_id"] == "1" and row["system"] == "LATTICE")
    row["mean_f1_percent"] = str(float(row["mean_f1_percent"]) + 0.5)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    checksums = read_json(folder / "checksums.json")
    checksums[path.name] = sha256(path)
    write_json(folder / "checksums.json", checksums)
    with pytest.raises(ValueError, match="interval differs from the recorded means"):
        verify(folder)
