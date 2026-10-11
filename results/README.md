# Paper results

These files contain the recorded experiment results used in the manuscript.
F1 values and differences use percentage points. Main-table means and sample SDs
describe the three development-selected runs in the
[run registry](paper/run_registry.csv).

## Tables and figures

| Manuscript item | Files | Contents |
|---|---|---|
| Table 1 | [dataset_statistics.csv](paper/dataset_statistics.csv) | Split sizes, test-entity counts, and label inventories |
| Table 2 | [main_results.csv](paper/main_results.csv), [main_results_individual.csv](paper/main_results_individual.csv) | Means and sample SDs, with TP/FP/FN and F1 for every run |
| Table 3 | [inference_ablations.csv](paper/inference_ablations.csv) | Kyrgyz inference interventions at the full-system threshold |
| Table 4 | [training_draws.csv](paper/training_draws.csv) | Results on three additional Kyrgyz training subsets |
| Figure 3a | [training_draw_scores.csv](paper/training_draw_scores.csv) | Four systems: means and sample SDs for each additional training subset |
| Figure 3b | [training_draw_intervals.csv](paper/training_draw_intervals.csv) | Six paired intervals against the two Lattice controls |
| Table 5 | [deployment_costs.csv](paper/deployment_costs.csv) | Inference time, throughput, memory, parameters, and precision |
| Figure 2; Table S1 | [paired_intervals.csv](paper/paired_intervals.csv) | CVSF-minus-comparator differences and 95% paired intervals |
| Table S2 | [error_changes.csv](paper/error_changes.csv) | Recovered/lost true positives and removed/added false positives |
| Supplementary Section 2 | [errors_summary.csv](paper/errors_summary.csv), [category_net_changes.csv](paper/category_net_changes.csv), [per_type_changes.csv](paper/per_type_changes.csv) | Error counts and changes by category or entity type |
| Table S3 | [uyghur_selected_runs.csv](paper/uyghur_selected_runs.csv) | Per-run Uyghur comparisons with external baselines |
| Table S4 | [uyghur_label_mapping.csv](paper/uyghur_label_mapping.csv) | All ten native types mapped to the five-type task |

## Experiment records

| File | Contents |
|---|---|
| [run_registry.csv](paper/run_registry.csv) | Component seeds and historical display identifiers |
| [internal_selection.csv](paper/internal_selection.csv) | Operating points for four languages and the three additional Kyrgyz draws |
| [external_selection.csv](paper/external_selection.csv) | Selected external checkpoint epochs and thresholds |
| [evaluation_order.json](paper/evaluation_order.json) | Selection and prediction completion times in each external evaluation batch |
| [training_cost_records.csv](paper/training_cost_records.csv) | Recorded training times, fit counts, and unmeasured stages |
| [uyghur_dataset_audit.json](paper/uyghur_dataset_audit.json) | Mapping, duplicate-record, and exact-text separation checks |
| [uyghur_errors_individual.csv](paper/uyghur_errors_individual.csv), [uyghur_error_changes_individual.csv](paper/uyghur_error_changes_individual.csv) | Per-run Uyghur error analysis |
| [checksums.json](paper/checksums.json) | SHA256 hashes of all result CSV files |

Fields named `source` and checkpoint references such as `fits/.../best` identify
the retained experiment archive. Those historical files are not included in this
clone. The tables above link to the files distributed here.

## Read the comparisons

The main table contains four languages, nine systems, and three runs: 108 count
rows and 36 means. The 24 external fits ran in two batches: 18 for Kyrgyz,
Turkish, and Mongolian, followed by six for Uyghur. Each batch completed selection
and prediction before scoring.

Means average run-level F1 without pooling entity counts. Some components were
reused, so the SDs do not describe independent full-pipeline retraining.
Paired intervals use 10,000 shared sentence resamples and ordinary, unadjusted
95% percentiles. They condition on the fitted models, training subsets, and
development choices; they exclude retraining, language, and document-level uncertainty.

CVSF has the highest mean F1 on all four datasets, but both Uyghur comparisons
with external models have intervals that include zero. The test corpora had also
been used in earlier development. Supplementary Section 4.5 explains that history;
the current intervals describe individual dataset comparisons.

Inference ablations retain the full fusion threshold. Their POOL-only and
Lattice-only rows therefore differ from separately selected systems in Table 2.
Additional training subsets can overlap and use partly different operating-point
rules from the original subset. POOL contrasts on those subsets are descriptive.

Error analysis covers all four languages. Timing measurements cover Kyrgyz,
Turkish, and Mongolian, under the conditions in the
[reproduction guide](../docs/reproduction.md#timing-and-training-records).

## Verify the results

```bash
cvsf results
cvsf results --markdown outputs/main_results.md
```

The verifier checks hashes and table coverage, recomputes main-table F1, means,
and sample SDs, and cross-checks the training-subset scores against their paired
differences. Displayed main-table values use three decimal places; calculations
retain the precision stored in the CSV files.

[reference_checks.json](reference_checks.json) records earlier comparisons with
the original experiment files. Unpublished reference code is identified by its
role and SHA256 hash. The [validation notes](../docs/validation.md) explain the
historical record and the checks available in this repository.
