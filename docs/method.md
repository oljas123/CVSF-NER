# Method and implementation

## From tokens to entities

The input is a tokenized sentence. The BIO model predicts a distribution over BIO labels at the first subword of each token. The candidate generator builds typed, half-open spans `(start, end, type)` from five sources:

1. Entities decoded from the BIO argmax sequence.
2. Local perturbations of those entity boundaries.
3. Paths through the three highest-scoring token labels.
4. Entity surfaces observed in the training split.
5. Short n-grams with token-level entity support.

Top-label paths, memory matches, and n-grams use a maximum width of eight words. BIO entities and their local boundary perturbations can be longer. The memory and n-gram generators each retain up to three types per boundary. Their union with the other sources can contain more than three types. Duplicate typed proposals are merged, with their source memberships retained. Candidate creation does not read the sentence's gold labels.

The implementation is in [candidates.py](../src/cvsf/candidates.py). `proposal_options()` contains the common settings used across languages.

## OOF supervision

Five BIO models each train on four folds and predict the remaining fold. Each model's surface memory uses only its training rows. Lattice and POOL learn from the combined candidates across all five held-out folds. OOF thus refers to exclusion from the proposing tagger's training data; both second-stage models train on these candidate examples.

Each typed candidate receives a binary target for exact boundary and type correctness. Each unique boundary receives a POOL target: its gold entity type, or `NONE` when the boundary is not a gold entity. The gold annotations determine targets after the boundary inventory has been built.

The final BIO proposer trains on all 500 training records. At inference, its candidates use surface memory from those same 500 records. The five OOF encoders are training resources; prediction uses the final BIO proposer and POOL encoder.

## Lattice score

Lattice uses 25 numeric features and two categorical features (`pred_type`, `source_signature`). The numeric features describe span length and position, candidate-source membership, BIO confidence, margins, entropy, expected BIO-label support, and nearby entity support. The `source_external_memory` feature is retained in the recorded feature order and is zero in these experiments.

The classifier is an ExtraTrees ensemble with 200 trees, minimum leaf size 2, and balanced class weights. These settings came from earlier development. The Kyrgyz scorer comparison used a sentence-level 80/20 split of OOF examples, followed by refitting on all candidates. The other languages retained the established scorer settings. The public runner fits this selected ExtraTrees model directly.

Numeric features receive median imputation; categorical features use one-hot encoding. [features.py](../src/cvsf/features.py) defines their order, and [lattice.py](../src/cvsf/lattice.py) implements fitting and scoring.

## Contextual verifier (POOL)

POOL reads the proposed boundaries with a second XLM-R-large encoder. For each span it concatenates the first subword state, last subword state, mean state inside the span, and mean state over the sentence's content subwords. A dropout layer and linear head produce a distribution over `NONE` and the corpus entity types.

Inputs pack at most 64 boundaries. Each boundary appends a marker pair with position IDs tied to its endpoints. Content tokens attend to content tokens; each marker pair can read the content and its own pair. POOL's classifier uses the four content-based representations described above. Since content tokens cannot attend to markers, packing does not introduce marker-mediated interactions between candidate representations. [packets.py](../src/cvsf/packets.py) defines the mask and [neural.py](../src/cvsf/neural.py) defines the forward pass.

The training loss sums cross-entropy across every candidate in an effective batch of eight sentences, then divides by the batch's total candidate count. The final partial batch is retained. All candidate packs in the batch contribute before the optimizer step.

## Fusion and decoding

For a proposed boundary and entity type, CVSF averages the POOL type probability and Lattice correctness probability with equal weight. Missing typed Lattice proposals receive a score of zero. It selects the highest-scoring entity type at each boundary, then solves weighted interval scheduling to obtain non-overlapping spans.

For candidate probability `p` and threshold `t`, the interval weight is `logit(p) - logit(t)`. Probabilities are clipped to `[10^-6, 1 - 10^-6]`. Only positive weights are eligible. Intervals are ordered by end, start, and decreasing probability. An exact dynamic-programming tie keeps the previous solution.

The threshold grid is 0.05 through 0.95 in steps of 0.05. Internal-system selection maximizes Dev-S F1, then precision, then fewer predicted entities, then the higher threshold. External GLiNER selection uses F1 and then the lower threshold. External checkpoint ties retain the earlier epoch.

The original Kyrgyz draw also includes the previously fixed FILTER, Lattice, and dual-BIO Lattice options. They are recorded in `configs/kyrgyz.json` and already lie on the same grid. Data preparation records the draw in `protocol.json`; selection includes these options for draw 2026 and uses the ordinary grid for additional draws. [internal_selection.csv](../results/paper/internal_selection.csv) lists the final choices for every released internal run.

## Internal controls

| Output name | Meaning |
|---|---|
| `BIO_RAW` | BIO argmax entities |
| `FILTER` | BIO entities with a selected confidence field and threshold |
| `LATTICE` | Expanded candidates scored by the tree classifier |
| `POOL` | Expanded candidate boundaries scored by POOL |
| `FUSION` | Equal-weight Lattice/POOL fusion |
| `PAIR_ENTITY_FILTER` | Entity-level averaging from two BIO models, with the single-model filter available during selection |
| `PAIR_TOKEN_FILTER` | Average BIO token probabilities, then filter decoded entities |
| `PAIR_TOKEN_LATTICE` | Average BIO token probabilities, then generate and score candidates |

The dual BIO controls use the primary seed and a partner seed offset by 2000. They average probabilities rather than logits. The Lattice classifier remains the one fitted from the primary BIO source's OOF candidates.

`PAIR_ENTITY_FILTER` is retained for comparison with earlier experiments. Table 2 reports the two token-level dual BIO controls.

## Evaluation

An entity is correct only when both token boundaries and type match. Precision, recall, and F1 are micro-averaged over the dataset. Invalid BIO `I-` transitions start a new predicted entity; reference labels are validated as IOB2. The optional three-member consensus requires at least two identical typed spans. The paper instead averages the scores of individual runs.

GLiNER's character predictions are converted to token spans using the offsets in the space-joined input. The pinned processor may append a terminal period; original token coordinates stay unchanged. The adapter checks this condition before prediction. Its flat output uses confidence-first overlap resolution across types. A prediction with an endpoint inside a token or in appended punctuation remains a false positive in the metric. Consensus votes on these unaligned character spans separately and preserves a span supported by at least two members. See [external.py](../src/cvsf/external.py).
