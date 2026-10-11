"""Validate the released count tables and render the main result table."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .io import read_json, sha256

LANGUAGES = dict(kyrgyz="Kyrgyz", turkishwiki="Turkish", mongolian="Mongolian", uyghur_bilegal="Uyghur")
SYSTEM_NAMES = {
    "BIO_RAW": "BIO tagger",
    "FILTER": "BIO + Filter",
    "LATTICE": "BIO + Lattice",
    "PAIR_TOKEN_FILTER": "Dual BIO + token Filter",
    "PAIR_TOKEN_LATTICE": "Dual BIO + token Lattice",
    "POOL": "POOL",
    "spanmarker_xlmr_large": "SpanMarker (XLM-R-large)",
    "gliner25_multi": "GLiNER2.5 multilingual",
    "FUSION": "CVSF (FUSION)",
}


def require_keys(rows, columns, expected, name):
    observed = [tuple(row[column] for column in columns) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"Incomplete or repeated language/system/run coverage in {name}")


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_training_draws(folder):
    rows = csv_rows(folder / "training_draw_scores.csv")
    systems = ("FUSION", "POOL", "LATTICE", "PAIR_TOKEN_LATTICE")
    draws = ("1", "2", "3")
    require_keys(rows, ("draw_id", "system"), {(d, s) for d in draws for s in systems}, "training_draw_scores.csv")
    lookup = {(row["draw_id"], row["system"]): row for row in rows}
    for row in rows:
        mean, sd = float(row["mean_f1_percent"]), float(row["sample_sd_f1_points"])
        if not (0 <= mean <= 100 and math.isfinite(sd) and sd >= 0 and int(row["members"]) == 3):
            raise ValueError(f"Invalid training-subset score: {row}")
    summaries = csv_rows(folder / "training_draws.csv")
    require_keys(summaries, ("draw_id",), {(d,) for d in draws}, "training_draws.csv")
    for row in summaries:
        draw = row["draw_id"]
        for system, field in (
            ("FUSION", "fusion_f1_mean"),
            ("POOL", "pool_f1_mean"),
            ("PAIR_TOKEN_LATTICE", "dual_bio_lattice_f1_mean"),
        ):
            if not math.isclose(float(lookup[(draw, system)]["mean_f1_percent"]), float(row[field]), abs_tol=1e-10):
                raise ValueError(f"Training-subset mean differs from Table 4: {draw}, {system}")
        fusion = lookup[(draw, "FUSION")]
        if not math.isclose(float(fusion["sample_sd_f1_points"]), float(row["fusion_f1_sample_sd"]), abs_tol=1e-10):
            raise ValueError(f"Training-subset SD differs from Table 4: {draw}")
        for comparator, field in (
            ("POOL", "fusion_minus_pool"),
            ("PAIR_TOKEN_LATTICE", "fusion_minus_dual_bio_lattice"),
        ):
            difference = float(fusion["mean_f1_percent"]) - float(lookup[(draw, comparator)]["mean_f1_percent"])
            if not math.isclose(difference, float(row[field]), abs_tol=1e-10):
                raise ValueError(f"Training-subset difference differs from Table 4: {draw}, {comparator}")
    intervals = csv_rows(folder / "training_draw_intervals.csv")
    require_keys(
        intervals,
        ("draw_id", "comparator"),
        {(d, s) for d in draws for s in ("LATTICE", "PAIR_TOKEN_LATTICE")},
        "training_draw_intervals.csv",
    )
    for row in intervals:
        draw, comparator = row["draw_id"], row["comparator"]
        difference = float(lookup[(draw, "FUSION")]["mean_f1_percent"]) - float(
            lookup[(draw, comparator)]["mean_f1_percent"]
        )
        if not math.isclose(difference, float(row["difference_f1_points"]), abs_tol=1e-10):
            raise ValueError(f"Training-subset interval differs from the recorded means: {draw}, {comparator}")
        lower, upper = float(row["lower_95_f1_points"]), float(row["upper_95_f1_points"])
        if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
            raise ValueError(f"Invalid training-subset interval: {row}")
    return len(rows)


def verify(folder):
    folder = Path(folder)
    checksums = read_json(folder / "checksums.json")
    if set(checksums) != {path.name for path in folder.glob("*.csv")}:
        raise ValueError("Result CSV inventory differs from checksums.json")
    for name, expected in checksums.items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Result export changed: {name}")
    individuals = csv_rows(folder / "main_results_individual.csv")
    require_keys(
        individuals,
        ("language", "system", "run_id"),
        {(language, system, run) for language in LANGUAGES for system in SYSTEM_NAMES for run in ("1", "2", "3")},
        "main_results_individual.csv",
    )
    groups = defaultdict(list)
    for row in individuals:
        tp, fp, fn = (int(row[name]) for name in ("tp", "fp", "fn"))
        if min(tp, fp, fn) < 0:
            raise ValueError(f"Negative entity count: {row}")
        computed = 200 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        if not math.isclose(computed, float(row["f1_percent"]), abs_tol=1e-10):
            raise ValueError(f"Count/F1 mismatch: {row}")
        groups[(row["language"], row["system"])].append(computed)
    summaries = csv_rows(folder / "main_results.csv")
    require_keys(
        summaries,
        ("language", "system"),
        {(language, system) for language in LANGUAGES for system in SYSTEM_NAMES},
        "main_results.csv",
    )
    for row in summaries:
        values = groups[(row["language"], row["system"])]
        if len(values) != 3 or not math.isclose(statistics.mean(values), float(row["f1_mean"]), abs_tol=1e-10):
            raise ValueError(f"Run mean mismatch: {row}")
        if not math.isclose(statistics.stdev(values), float(row["f1_sample_sd"]), abs_tol=1e-10):
            raise ValueError(f"Run SD mismatch: {row}")
    externals = csv_rows(folder / "external_selection.csv")
    require_keys(
        externals,
        ("language", "system", "run_id"),
        {
            (language, system, run)
            for language in LANGUAGES
            for system in ("spanmarker_xlmr_large", "gliner25_multi")
            for run in ("1", "2", "3")
        },
        "external_selection.csv",
    )
    intervals = csv_rows(folder / "paired_intervals.csv")
    comparators = (
        "POOL",
        "LATTICE",
        "PAIR_TOKEN_FILTER",
        "PAIR_TOKEN_LATTICE",
        "spanmarker_xlmr_large",
        "gliner25_multi",
    )
    require_keys(
        intervals,
        ("language", "comparator"),
        {(language, system) for language in LANGUAGES for system in comparators},
        "paired_intervals.csv",
    )
    for row in intervals:
        language, comparator = row["language"], row["comparator"]
        difference = statistics.mean(groups[(language, "FUSION")]) - statistics.mean(groups[(language, comparator)])
        if not math.isclose(difference, float(row["difference_f1_points"]), abs_tol=1e-10):
            raise ValueError(f"Paired difference does not match member means: {row}")
        if not float(row["lower_95"]) <= float(row["upper_95"]) or int(row["resamples"]) != 10000:
            raise ValueError(f"Invalid interval export: {row}")
    return dict(
        individual_rows=len(individuals),
        group_rows=len(summaries),
        external_rows=len(externals),
        interval_rows=len(intervals),
        csv_files=len(read_json(folder / "checksums.json")),
        training_draw_score_rows=verify_training_draws(folder),
    )


def main_table(folder):
    folder = Path(folder)
    verify(folder)
    rows = csv_rows(folder / "main_results.csv")
    lookup = {(row["language"], row["system"]): row for row in rows}
    lines = ["| System | " + " | ".join(LANGUAGES.values()) + " |", "|---|" + "---:|" * len(LANGUAGES)]
    for system, label in SYSTEM_NAMES.items():
        cells = [label]
        for language in LANGUAGES:
            row = lookup[(language, system)]
            cells.append(f"{float(row['f1_mean']):.3f} ± {float(row['f1_sample_sd']):.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
