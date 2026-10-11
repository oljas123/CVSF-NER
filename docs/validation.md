# Validation

The public tests exercise numerical operations, data preparation, model adapters,
and a complete CPU training run. Separate checks compared the implementation with
retained experiment files. Those comparisons are recorded in
[reference_checks.json](../results/reference_checks.json), with the source hashes
used at the time.

The JSON is a historical record dated by `checked_utc`. Its reference source IDs
describe archived implementations that are not distributed with this repository;
they are not file paths. `released_python_sha256` contains repository-relative
paths and hashes from that check, so later source revisions may differ. Test
counts also refer to that historical run. Use the commands below to check the
current checkout.

| Check | Coverage |
|---|---|
| Main results | Four languages, 108 TP/FP/FN rows, 36 means, 24 external selections, and 24 paired differences |
| Training-subset results | Twelve system/subset summaries, cross-checked against the means and FUSION SDs in Table 4 and six paired differences |
| Corpus reconstruction | Seven manifests restored from pinned raw files, with record/token hashes and exact-text separation checked |
| Candidates and features | Twelve saved BIO outputs per language; typed candidates and all 27 feature fields matched the original code |
| POOL | Identical initialization, forward logits, and parameters after two updates using a small encoder |
| Fusion and decoding | Thirty-two saved test sentences per language for Turkish, Mongolian, and Uyghur, at five thresholds |
| Kyrgyz selection | Recorded operating points recovered for eight systems and three runs on 327 Dev-S sentences |
| Kyrgyz predictions | All 3,569 sentence-level entity sets matched the archived outputs for eight systems and three runs |
| SpanMarker reload | Nested eager attention, parameters, logits, and predictions preserved across a small-model save/reload |
| GLiNER offsets | Original coordinates preserved when the pinned processor appends punctuation; other coordinate changes rejected |
| CPU integration | Five OOF fits, both BIO models, Lattice, POOL, selection, and eight prediction outputs |

The saved-score and small-model checks do not include a new XLM-R-large training
run. Historical trained checkpoints and score archives are not distributed here;
the training commands create new runs from the recorded inputs and settings.

## Internal tests

After installing the internal environment, run:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src tests
cvsf results
```

The model tests construct small encoders locally and need no model downloads.
Corpus reconstruction can also run offline with `cvsf prepare --offline` once the
pinned raw files are cached.

## External tests

In the separate external environment, run the adapter and checkpoint tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_external.py tests/test_external_neural.py
```

The SpanMarker test checks logits as well as decoded predictions after reloading.
This catches attention-mode changes even when a small model predicts no entities.

## Result tables

`cvsf results` recomputes the main table from individual entity counts and checks
the additional training-subset results. The verifier also compares their means
with the paired differences, even when the file checksums have been updated.
