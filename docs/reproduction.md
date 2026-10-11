# Reproduction guide

Run these commands from the repository root after completing the
[installation](../README.md#install). Multiline examples use Bash syntax; in
PowerShell, join their lines into a single command.

## Environments

The internal BIO and POOL models use `requirements/internal.txt`, which includes
Transformers 5.9.0, PyTorch 2.11.0, and Accelerate 1.13.0. The recorded runtime used
Python 3.10.20, CUDA 13.0, and an RTX 5080 with 16 GB VRAM.

SpanMarker and GLiNER2.5 use Transformers 4.57.6 in a separate environment:

```bash
python -m venv .venv-external
source .venv-external/bin/activate
python -m pip install -r requirements/external.txt
python -m pip install -e .
```

On Windows, activate it with `.venv-external\Scripts\Activate.ps1`.
The external requirements pin SpanMarker 1.8.1 and the source commit for GLiNER2
package 2.1.0. Language configs pin the pretrained model revisions. Downloads use
the Hugging Face cache, which can be relocated with `HF_HOME`.

## Run mapping

| Run | Primary BIO / OOF / Lattice seed | Partner BIO seed | POOL seed | External seed |
|---|---:|---:|---:|---:|
| Run1 | 2037 | 4037 | 2037 | 2037 |
| Run2 | 2038 | 4038 | 2038 | 2038 |
| Run3 | 2036 | 4036 | 2040 | 2040 |

All four languages use this mapping. Run3 appears as 2039 in some archived
filenames; its POOL training seed is 2040. The
[run registry](../results/paper/run_registry.csv) records this distinction.

Runs 1–3 are configurations selected during development, with some component
reuse. Their sample SDs describe these configurations, not independent training
of the full pipeline. Main-table means average the three run-level F1 scores.

## Internal training

Prepare the dataset once, then use a separate directory for each run:

```bash
cvsf prepare --manifest manifests/kyrgyz.json --output data/kyrgyz
cvsf train --config configs/kyrgyz.json --data data/kyrgyz --run Run1 --output runs/kyrgyz/Run1
cvsf select --model runs/kyrgyz/Run1 --data data/kyrgyz
cvsf predict --model runs/kyrgyz/Run1 --input data/kyrgyz/test.jsonl --output outputs/kyrgyz/Run1
```

Repeat the last three commands for `Run2` and `Run3`. Replace `kyrgyz` with
`turkishwiki`, `mongolian`, or `uyghur_bilegal` for the other corpora.
Use `--device cpu` for small experiments. The `--without-controls` option skips
the partner BIO fit and produces the five single-proposer systems.

Each complete run fits five OOF BIO models, the primary and partner BIO models,
the ExtraTrees scorer, and POOL. BIO and POOL train for 20 epochs and retain their
final weights. Both use AdamW, weight decay 0.01, gradient clipping at 1.0, and
linear learning-rate decay without warm-up.

BIO uses batches of two sentences with four accumulation steps. Its learning
rate is 2e-5 for Kyrgyz and 3e-5 elsewhere. Kyrgyz BIO uses BF16 autocast; the
other BIO fits use FP32.

POOL accumulates eight sentences through candidate packs of at most 64 boundaries.
It divides the summed cross-entropy by the candidate count in that update,
including the final partial batch. Encoder and head learning rates are 2e-5 and
1e-4. POOL trains with BF16 autocast; all neural weights remain FP32.

`select` chooses operating points on Dev-S and records metrics for both Dev-S
and Dev-C. Prediction uses those saved choices and reads only sentence tokens.
BIO inference uses BF16 autocast for Kyrgyz and FP32 elsewhere; POOL inference uses FP32.

Completed phases are reused when configuration and input hashes match. Use a new
output directory after changing either. An interrupted fit restarts from its
beginning; the runner does not restore optimizer state. Keeping all three internal
runs takes roughly 60 GB before external models and large prediction caches.

`--max-steps` shortens BIO or external training for debugging. Omit it for the
paper settings. `cvsf smoke` also shortens POOL training and builds a small local
encoder for a complete CPU check.

## Output files

| File or directory | Contents |
|---|---|
| `inputs.json` | Configuration, seeds, and input hashes |
| `bio/fold_0` … `bio/fold_4` | OOF proposer checkpoints and held-out predictions |
| `bio/primary`, `bio/partner` | BIO checkpoints fitted on all training records |
| `oof_typed_candidates.jsonl` | Candidate features and binary correctness targets |
| `pool_training_candidates.jsonl` | Candidate boundaries and POOL targets |
| `lattice.joblib`, `memory.json` | Fitted tree pipeline and training surface counts |
| `pool/` | POOL configuration and safetensors weights |
| `selection.json`, `selection_curves.json` | Selected operating points and grid evaluations |
| `development_metrics.json` | Exact-span metrics on Dev-S and Dev-C |

`predict` writes one JSONL file per system. Its input model directory must contain
the primary BIO, POOL, Lattice, memory, schema, configuration, and selection files.
The partner BIO is also needed when controls are enabled. External prediction
requires the run metadata, selection file, and `best/` checkpoint.

The training commands create these files from the pinned encoders and corpus
sources. Historical trained checkpoints are not included in this repository.
Load serialized model files only from a trusted run. Generated data, weights, and
outputs are excluded from Git.

## External baselines

Activate the external environment and reuse the prepared data:

```bash
cvsf external-train --config configs/kyrgyz.json --data data/kyrgyz --arm spanmarker --run Run1 --output runs/external/kyrgyz/spanmarker/Run1
cvsf external-predict --model runs/external/kyrgyz/spanmarker/Run1 --input data/kyrgyz/test.jsonl --output outputs/external/kyrgyz/spanmarker_Run1.jsonl
cvsf evaluate --gold data/kyrgyz/test.jsonl --predictions outputs/external/kyrgyz/spanmarker_Run1.jsonl
```

For GLiNER2.5, use `--arm gliner` and separate run and output paths. Repeat for all
three runs and four languages. Both methods train for 20 epochs, evaluate Dev-S
after each epoch, and retain the highest strict entity F1. Checkpoint ties retain
the earlier epoch. GLiNER also selects a threshold from 0.05 to 0.95 in steps of
0.05, with the lower threshold breaking an F1 tie.

SpanMarker uses XLM-R-large, learning rate 2e-5, warm-up ratio 0.1, and eight
accumulated training examples. Its maximum input length is 512 and marker capacity
is 64. Maximum entity widths, measured from training and Dev-S, are 22, 14, 8,
and 47 words for Kyrgyz, Turkish, Mongolian, and Uyghur.

SpanMarker retains its native loss and decoding. The adapter normalizes gradient
accumulation explicitly because the native loss does not consume the trainer's
summed-item normalization argument. Span packing can turn one sentence into
multiple model inputs, so matching accumulation settings does not imply equal computation.

GLiNER2.5 uses encoder/task learning rates of 1e-5/5e-4, warm-up ratio 0.1,
eight-sentence accumulation, maximum length 4,096, and zero synthetic-label probability.
Both external methods train with BF16 autocast and FP32 weights. Development
selection and reloaded-checkpoint inference use FP32. SpanMarker restores eager
attention in its nested encoder when reloading.

## Evaluation and paired comparisons

Evaluate a single prediction file with:

```bash
cvsf evaluate --gold data/kyrgyz/test.jsonl --predictions outputs/kyrgyz/Run1/FUSION.jsonl --output outputs/kyrgyz/Run1/metrics.json
```

To compare the three-run mean for FUSION against POOL on Kyrgyz:

```bash
cvsf bootstrap \
  --gold data/kyrgyz/test.jsonl \
  --left outputs/kyrgyz/Run1/FUSION.jsonl outputs/kyrgyz/Run2/FUSION.jsonl outputs/kyrgyz/Run3/FUSION.jsonl \
  --right outputs/kyrgyz/Run1/POOL.jsonl outputs/kyrgyz/Run2/POOL.jsonl outputs/kyrgyz/Run3/POOL.jsonl \
  --seed 11005 --index-dtype int32 --repetitions 10000 \
  --output outputs/kyrgyz/fusion_vs_pool.json
```

Each shared sentence resample recomputes F1 within each run, then averages the
paired differences. Keep files in matching Run1–Run3 order. Passing one file per
side instead compares one fitted run. Evaluation metrics are fractions in
`[0, 1]`; bootstrap differences and released tables use F1 percentage points.

| Language | Internal seed (`int32`) | SpanMarker seed (`int64`) | GLiNER seed (`int64`) |
|---|---:|---:|---:|
| Kyrgyz | 11005 | 19019 | 19020 |
| Turkish | 16005 | 19029 | 19030 |
| Mongolian | 16006 | 19039 | 19040 |
| Uyghur | 16007 | 19049 | 19050 |

All paper comparisons use 10,000 resamples. The 95% percentile intervals are
unadjusted and conditional on the fitted models, training subsets, and development
choices. They do not include uncertainty from retraining, language sampling, or
document clustering. These test sets had also been used during earlier development;
Supplementary Section 4.5 describes that history and its interpretation.

The optional `consensus` command performs two-of-three exact-span voting:

```bash
cvsf consensus --predictions outputs/kyrgyz/Run1/FUSION.jsonl outputs/kyrgyz/Run2/FUSION.jsonl outputs/kyrgyz/Run3/FUSION.jsonl --output outputs/kyrgyz/consensus.jsonl
```

Consensus produces a separate prediction set; the paper reports averages of
individual run scores.

## Inference ablations

```bash
cvsf ablate --model runs/kyrgyz/Run1 --input data/kyrgyz/test.jsonl --output outputs/kyrgyz/Run1/ablations
```

Repeat for all three runs, evaluate the files, and compare each variant with
FUSION using the paired bootstrap. The command retains the fitted models and
FUSION threshold. It writes POOL-only and Lattice-only scores, BIO-boundary and
proposal-type restrictions, greedy decoding, and the full system.

These are inference interventions without retraining or threshold reselection.
The POOL-only and Lattice-only scores therefore differ from the independently
selected systems in the main table.

## Additional training subsets

The three additional Kyrgyz subsets contain 500 sentences each and use draw
identifiers 31001, 31002, and 31003. To restore and train the first:

```bash
cvsf prepare --manifest manifests/kyrgyz_draw31001.json --output data/kyrgyz_draw31001
cvsf train --config configs/kyrgyz.json --data data/kyrgyz_draw31001 --run Run1 --output runs/kyrgyz_draw31001/Run1
```

Select and predict with that data directory, then repeat for all runs and draws.
Each manifest records its own training membership and OOF folds. Development and
test membership stay fixed. The subsets can overlap and share the selected run
configurations, so they are not independent datasets.

The original draw 2026 includes previously fixed options during selection:

| Run | FILTER field | FILTER threshold | Lattice and dual-BIO Lattice threshold |
|---|---|---:|---:|
| Run1 | `candidate_score` | 0.70 | 0.75 |
| Run2 | `min_expected_prob` | 0.60 | 0.75 |
| Run3 | `min_expected_prob` | 0.05 | 0.75 |

These options already lie on the 0.05-step grid; retaining them preserves the
original selection order. Additional draws use the grid without the extra entries.
The [selected operating points](../results/paper/internal_selection.csv) cover
all 21 internal runs, including the single-FILTER fallback for the entity-level control.
A new fit chooses its own Dev-S operating point.

## Timing and training records

[deployment_costs.csv](../results/paper/deployment_costs.csv) contains measurements
on 128 fixed sentences per language for Kyrgyz, Turkish, and Mongolian. Each run
used 16 warm-up sentences, four-sentence batches, and three timed repetitions.
POOL packs held at most 64 candidates. PyTorch and BLAS used four threads;
ExtraTrees used `n_jobs=-1`. GPU synchronization bracketed each measurement.

The machine had an RTX 5080, Ryzen 7 9800X3D, about 32 GB RAM, and driver 596.49.
Times include tokenization, candidate generation, neural inference, CPU tree
scoring, and decoding; model loading and file I/O are excluded. They measure
amortized batch processing rather than single-request latency.

The table first averages repetitions within a run, then reports mean ± run SD.
Peak allocated GPU memory is the maximum over all nine measurements and includes
resident neural weights. CPU, training, and other-process memory were not measured.
The CSV records each system's inference precision and Transformers version.

[training_cost_records.csv](../results/paper/training_cost_records.csv) contains
recorded component times. Internal totals omit stages listed in `missing_stages`;
external totals cover a complete 20-epoch fine-tuning job with development selection.
They do not measure the same training boundaries.

## Released results

```bash
cvsf results --markdown outputs/main_results.md
```

`results` checks CSV hashes and coverage, recomputes the main table from TP/FP/FN,
and checks the paired differences and training-subset results. See the
[result index](../results/README.md) for file mappings.

Retraining follows the recorded algorithm, splits, and settings. Device kernels
and library versions can still produce different fitted models and scores.
