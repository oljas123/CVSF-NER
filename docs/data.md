# Data preparation

## Sources

| Corpus | Source | Terms recorded with the experiment |
|---|---|---|
| KyrgyzNER | [Akyl-AI/KyrgyzNER](https://github.com/Akyl-AI/KyrgyzNER) | CC BY-NC-SA 4.0 |
| TurkishWikiNER | [Turkish-Wiki-NER-Dataset](https://github.com/turkish-nlp-suite/Turkish-Wiki-NER-Dataset) | CC BY-SA 4.0 |
| MongolianNER | [NER_v1.0.json.gz](https://github.com/tugstugi/mongolian-nlp/blob/465a9a5f01a810b25c45ac175be776fffbf6cd66/datasets/NER_v1.0.json.gz) | No dataset license declared in the source repository |
| BiLegalNERD (Uyghur) | [BiLegalNERD](https://github.com/PaErHaTi-DUTiR/BiLegalNER/tree/acbb0399196f2be9b816e49c6eca24a71b01ab55/BiLegalNERD) | Research use, as stated by the authors |

Download the corpus texts from their original sources using the preparation command below. This repository provides retrieval code and split manifests; each corpus retains its source terms. The pinned revisions, filenames, and SHA256 values are in [data_sources.json](../configs/data_sources.json).

## Restore the splits

```bash
cvsf prepare --manifest manifests/turkishwiki.json --output data/turkishwiki
```

The first invocation downloads the raw files into `data/raw/<dataset>/`. Later invocations reuse files with matching checksums. To prepare data without a network connection, place the files named in `data_sources.json` in that cache directory and add `--offline`.

Each manifest stores sentence IDs, `record_sha256` for normalized tokens plus BIO annotations, and a separate `token_sha256` for the token sequence alone, in experiment order. Hash payloads use UTF-8 JSON with `ensure_ascii=False` and separators `(',', ':')`. Manifests also store the training draw, exact OOF fold indices, Dev-S/Dev-C indices, and BIO label order. Data preparation matches both hashes, checks OOF coverage and exact-text separation, and writes the selected rows. A source or normalization mismatch raises an error with the affected split.

| Corpus | Training | Dev-S | Dev-C | Test | Types |
|---|---:|---:|---:|---:|---:|
| KyrgyzNER | 500 | 327 | 328 | 3,569 | 25 |
| TurkishWikiNER | 500 | 500 | 500 | 1,000 | 19 |
| MongolianNER | 500 | 457 | 458 | 1,017 | 4 |
| BiLegalNERD (Uyghur) | 500 | 160 | 161 | 318 | 5 |

The 500-record budget refers to training. Dev-S and Dev-C are additional labeled development data. Dev-S selects operating points and external checkpoints; Dev-C provides a separate development comparison.

## Corpus-specific processing

KyrgyzNER uses the cleaned experiment split. Exact token duplicates keep the first released occurrence, conflicting annotation groups are removed, and released training data take precedence over overlapping released test text. The cleaned training pool is divided into training and development with the recorded seed 2026. The 500-sentence subset and its five OOF folds retain the original Python-random order. The public label order is kept exactly, including the position of `O`.

TurkishWikiNER keeps the released train/development/test roles. Its 500 training sentences are a fixed sample. The underscore-to-hyphen normalization of tags follows the experiment preprocessing; use the supplied schema when inspecting label names.

MongolianNER starts from Doccano character annotations. Tokenization splits whitespace-delimited tokens at annotated character boundaries so that each source span remains representable. The train/development/test membership follows the authors' two-stage split: 10% test, then 10% of the remainder for development, both with random state 1. The exact resulting membership is recorded in the manifest.

BiLegalNERD (Uyghur) maps `NHCS` and `NHVI` to `PER`, `NS` to `LOC`, `NO` to `ORG`, and `NT` to `TIME`. `NCGV`, `NCSM`, `NASI`, `NCSP`, and `NATS` map to `MISC`. The 500 training records contain 485 distinct token sequences. Duplicate text stays within one OOF fold. Six training-overlapping development records are removed; the test manifest contains 318 distinct sentence texts.

The five-type mapping covers all ten native types and preserves all 15,616 entities before duplicate and overlap exclusions. Scores therefore describe a different task from the native ten-type benchmark.

Nine duplicate-text groups account for 24 training records, with consistent native and mapped annotations. All 500 records remain as separate training instances, without inverse-multiplicity weighting. Their contribution follows each model's loss normalization; POOL averages over candidates within each accumulated update. Internal and external systems use the same mapped records. [uyghur_label_mapping.csv](../results/paper/uyghur_label_mapping.csv) gives the full mapping, and [uyghur_dataset_audit.json](../results/paper/uyghur_dataset_audit.json) records duplicate counts, mapping checks, and zero exact-text intersections among train, Dev-S, Dev-C, and test. These checks do not assess near-duplicates or document identity.

Turkish, Mongolian, and Uyghur folds group exact token sequences and allocate groups to the smallest fold, using the recorded hash order to break ties. Their Dev-S/Dev-C partition seeds are 16001, 16002, and 16003. All these indices are included, so reproduction does not depend on re-running a split algorithm.

## JSONL format

Training and evaluation files contain one sentence per line:

```json
{"sent_id":"example-1","tokens":["Ada","visited","Luma","."],"tags_bio":["B-PER","O","B-LOC","O"]}
```

This example is synthetic. Sentence IDs are unique within a split, and token and label arrays have equal length. Prediction input needs only `sent_id` and `tokens`. The prediction command constructs token-only records before invoking any model.

Outputs use zero-based, half-open token offsets:

```json
{"sent_id":"example-1","entities":[[0,1,"PER"],[2,3,"LOC"]]}
```

External GLiNER outputs may also contain `unaligned_entities`, expressed as character offsets in the model's normalized, space-joined input. Evaluation counts each such output as a false positive.

## New datasets

Provide `train.jsonl`, `valid.jsonl`, `label_schema.json`, `folds.json`, and `partition.json` in the same format as a prepared corpus. `folds.json` is a list of five objects containing `train` and `heldout` zero-based indices into `train.jsonl`. `partition.json` contains disjoint `selection` and `comparison` indices into `valid.jsonl`. Keep duplicate token sequences in the same fold and keep training, development, and test text disjoint.

Both training entry points check fold coverage, disjoint indices, same-text grouping, and development partitions before loading an encoder. Custom datasets use the ordinary selection grid. `prepare` additionally writes `protocol.json` so the original Kyrgyz draw can restore its recorded eligible options.

Copy a language config and set the encoder, ordered schema, maximum input length, and training recipe for your experiment. The supplied paper configurations remain useful as references. Long inputs are rejected unless truncation is enabled explicitly; the Kyrgyz experiment uses the recorded 256-subword truncation policy.
