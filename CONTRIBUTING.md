# Contributing

Open an issue for a bug, a reproduction question, or a proposed protocol change.
Small fixes can go directly to a pull request.

## Development setup

In the internal environment, install the development tools and run:

```bash
python -m pip install -r requirements/internal.txt
python -m pip install -e ".[dev]"
python -m pytest -q -m "not integration"
python -m ruff check src tests
```

The complete CPU test trains small local models without downloading a corpus or encoder:

```bash
python -m pytest -q -m integration
```

External adapter changes also need the tests in the
[external environment](docs/validation.md#external-tests).

## Changes and tests

Keep each pull request focused. Describe the problem, the resulting behavior,
and the checks you ran. Add a regression test for changes to candidate generation,
alignment, fusion, selection, or evaluation.

Candidate generation must remain independent of a sentence's reference labels.
Keep split membership and label order explicit. Give a new experimental recipe
its own configuration and output directory so that existing results remain reproducible.

Use comments to explain numerical constraints or choices that are difficult to
infer from the code. Keep formatting-only changes separate from behavior changes.

## Bug reports

Include the command, configuration, OS, Python and package versions, device, and
relevant traceback. For a numerical mismatch, include the manifest hash and the
first stage where results diverge. A small synthetic example helps reproduce an
alignment or decoding error.

Do not attach credentials, private data, or large checkpoints. For a corpus-specific
issue, provide the source sentence ID and a description consistent with its terms.

## Results and documentation

Files in `results/paper/` record the manuscript experiments. Keep new experiment
outputs elsewhere and record their configuration, input hashes, counts, and
selection metadata. Update examples and guides whenever a command-line argument changes.

Contributions are distributed under this repository's MIT License.
