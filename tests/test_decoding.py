import itertools
import math

import numpy as np

from cvsf.decoding import decode, fuse, logit
from cvsf.metrics import entities, evaluate, vote
from cvsf.selection import select
from cvsf.ablations import ablate
from cvsf.metrics import member_bootstrap


def test_interval_decoder_matches_exhaustive_search():
    spans = [
        [0, 3, "A", 0.78],
        [0, 1, "A", 0.72],
        [1, 2, "B", 0.73],
        [2, 4, "A", 0.74],
        [3, 4, "B", 0.62],
        [4, 5, "A", 0.51],
    ]
    threshold = 0.55
    best = 0.0
    for size in range(len(spans) + 1):
        for subset in itertools.combinations(spans, size):
            ordered = sorted(subset)
            if all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:])):
                best = max(best, sum(logit(row[3]) - logit(threshold) for row in subset))
    chosen = decode(spans, threshold)
    actual = sum(logit(row[3]) - logit(threshold) for row in spans if tuple(row[:3]) in chosen)
    assert math.isclose(actual, best, abs_tol=1e-12)
    assert decode([[0, 1, "A", 0.5]], 0.5) == set()


def test_fusion_zero_fills_missing_typed_proposals():
    output = fuse([[0, 1]], [[0, 1, "A", 0.8]], np.log([[0.1, 0.3, 0.6]]), ["A", "B"])
    assert output["POOL"][0][:3] == [0, 1, "B"]
    assert output["FUSION"][0][:3] == [0, 1, "A"]
    assert math.isclose(output["FUSION"][0][3], 0.55)
    assert fuse([], [], np.empty((0, 3)), ["A", "B"]) == {"POOL": [], "FUSION": []}


def test_exact_span_metrics_and_voting():
    gold = [{(0, 2, "PER"), (3, 4, "LOC")}]
    prediction = [{(0, 1, "PER"), (3, 4, "LOC")}]
    scores = evaluate(prediction, gold)
    assert (scores["tp"], scores["fp"], scores["fn"], scores["f1"]) == (1, 1, 1, 0.5)
    assert entities(["I-PER", "I-PER", "O", "B-LOC"]) == [(0, 2, "PER"), (3, 4, "LOC")]
    assert vote([gold, prediction, gold]) == gold


def test_selection_uses_only_requested_sentences_and_retains_tie_rule():
    source = [dict(bio=[], FUSION=[[0, 1, "A", 0.91]]), dict(bio=[], FUSION=[[0, 1, "A", 0.2]])]
    decisions, _ = select(source, [{(0, 1, "A")}, set()], [0], ["FUSION"])
    changed, _ = select(source, [{(0, 1, "A")}, {(0, 1, "B")}], [0], ["FUSION"])
    assert decisions == changed
    assert decisions["FUSION"]["option"]["threshold"] == 0.9


def test_ablation_retains_full_system_and_restricts_boundaries():
    baseline = dict(lattice=[[0, 1, "A", 0.9], [1, 2, "A", 0.8]], bio=[dict(s=0, e=1)])
    logits = np.log([[0.1, 0.9], [0.2, 0.8]])
    result = ablate([[0, 1], [1, 2]], baseline, logits, ["A"], 0.5)
    assert result["FUSION"] == {(0, 1, "A"), (1, 2, "A")}
    assert result["BIO_BOUNDARIES_FIXED"] == {(0, 1, "A")}


def test_member_bootstrap_identical_systems_have_zero_interval():
    gold = [{(0, 1, "A")}, set()]
    members = [gold, [set(), set()], gold]
    result = member_bootstrap(members, members, gold, repetitions=100)
    assert result["delta_f1"] == result["lower"] == result["upper"] == 0.0
