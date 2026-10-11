# CVSF-NER

Code for **Candidate Verification and Score Fusion for Low-Resource Named Entity Recognition**.

CVSF expands the spans proposed by a BIO tagger, verifies them with a second
encoder, and combines the verifier and Lattice scores. Both scoring models learn
from out-of-fold (OOF) BIO predictions. A weighted interval decoder selects
non-overlapping entities from the scored candidates.

The experiments cover KyrgyzNER, TurkishWikiNER, MongolianNER, and the Uyghur
portion of BiLegalNERD, with 500 labeled training records per language and
additional development labels.

## Install

Use Python 3.10 or newer and run the examples from the repository root.

```bash
git clone https://github.com/oljas123/CVSF-NER.git
cd CVSF-NER
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/core.txt
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.

The core installation can verify the recorded results without downloading data
or models:

```bash
cvsf results
```

For neural training and inference, install the internal model dependencies:

```bash
python -m pip install -r requirements/internal.txt
cvsf smoke --output outputs/smoke
```

The smoke check runs the complete pipeline on CPU using synthetic sentences and
a small, locally initialized encoder. Its scores serve only to check the installation.

The experiments used PyTorch 2.11.0 with CUDA 13.0 on an RTX 5080 with 16 GB VRAM.
Use a PyTorch build suited to your hardware. SpanMarker and GLiNER2.5 need the
[separate external environment](docs/reproduction.md#environments).

## Reproduce a run

```bash
cvsf prepare --manifest manifests/kyrgyz.json --output data/kyrgyz
cvsf train --config configs/kyrgyz.json --data data/kyrgyz --run Run1 --output runs/kyrgyz/Run1
cvsf select --model runs/kyrgyz/Run1 --data data/kyrgyz
cvsf predict --model runs/kyrgyz/Run1 --input data/kyrgyz/test.jsonl --output outputs/kyrgyz/Run1
cvsf evaluate --gold data/kyrgyz/test.jsonl --predictions outputs/kyrgyz/Run1/FUSION.jsonl
```

Repeat the training, selection, prediction, and evaluation steps for `Run2` and
`Run3`. For another language, replace `kyrgyz` with `turkishwiki`, `mongolian`, or
`uyghur_bilegal` in the paths.

Data preparation downloads the pinned source files, checks their SHA256 hashes,
and restores the recorded splits and sentence order. Trained checkpoints are not
included; prediction uses the run directory produced by training and selection.

The [reproduction guide](docs/reproduction.md) covers all runs, external baselines,
paired comparisons, and resource use. The [data guide](docs/data.md) explains
corpus preparation and custom inputs; the [method guide](docs/method.md) describes
candidate generation, verification, and decoding.

## Results

Strict entity micro-F1 (%), reported as mean ± sample SD across three selected
run configurations. Values use the same three-decimal precision as Table 2.

| System | Kyrgyz | Turkish | Mongolian | Uyghur |
|---|---:|---:|---:|---:|
| BIO tagger | 66.117 ± 0.740 | 75.383 ± 0.170 | 85.664 ± 1.760 | 74.791 ± 0.357 |
| BIO + Filter | 68.729 ± 0.744 | 76.320 ± 0.384 | 86.330 ± 1.319 | 75.520 ± 0.675 |
| BIO + Lattice | 69.494 ± 0.577 | 75.736 ± 0.108 | 86.038 ± 1.106 | 76.239 ± 0.441 |
| Dual BIO + token Filter | 69.288 ± 0.168 | 76.439 ± 0.485 | 86.985 ± 0.650 | 76.203 ± 0.395 |
| Dual BIO + token Lattice | 69.243 ± 0.638 | 75.268 ± 0.301 | 86.190 ± 0.213 | 76.489 ± 0.459 |
| POOL | 69.413 ± 1.147 | 78.072 ± 0.524 | 87.501 ± 0.461 | 77.444 ± 0.809 |
| SpanMarker (XLM-R-large) | 69.099 ± 1.257 | 76.681 ± 0.236 | 86.746 ± 0.453 | 77.888 ± 0.403 |
| GLiNER2.5 multilingual | 67.718 ± 0.533 | 72.258 ± 1.034 | 76.736 ± 0.371 | 77.211 ± 0.401 |
| CVSF (FUSION) | 71.727 ± 0.461 | 78.359 ± 0.150 | 87.919 ± 0.525 | 78.114 ± 0.304 |

`FUSION` names the complete CVSF pipeline in output files. The
[run registry](results/paper/run_registry.csv) gives the component training seeds.
Some components were reused, so these SDs do not estimate variation from
independent training of the full pipeline.

The [result index](results/README.md) maps the exported data to the paper's tables
and figures. It also explains the paired intervals, including the Uyghur
comparisons with external models, whose intervals include zero.

## Repository layout

| Directory | Contents |
|---|---|
| `src/cvsf/` | Models, candidate generation, training, decoding, and command-line tools |
| `configs/` | Language settings and pinned data sources |
| `manifests/` | Split membership, OOF folds, and label order |
| `requirements/` | Core, internal, and external dependencies |
| `results/paper/` | Recorded counts, scores, intervals, and timing measurements |
| `tests/` | Numerical, data-alignment, adapter, and CPU integration tests |
| `docs/` | Method, data, reproduction, and validation guides |

## Contributing and citation

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and bug-report guidance.
The [validation notes](docs/validation.md) describe the public tests and comparisons
with the original experiment files.

[CITATION.cff](CITATION.cff) provides citation metadata. Publication details will
be added when the accompanying manuscript is published. Please also cite the
corpora and pretrained models used in your experiment.

The code uses the [MIT License](LICENSE). Corpus and model terms are documented
in the [data guide](docs/data.md) and [third-party notices](THIRD_PARTY.md).
