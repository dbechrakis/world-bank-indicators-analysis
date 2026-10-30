# Validation record

Review date: 2026-09-05 (UTC).

Executed both full analysis and SQL loading against all six committed CSVs; checked row grain and foreign keys.

Historical records are not labelled as freshly reproduced results.

## Refactor check — 2026-10-05

Moved data preparation and outlier statistics out of `scripts/world_bank_analysis.py` into the `wb_indicators` package and added 7 unit tests (country lookup, aggregate removal, long format, latest-year window, z-scores, IQR fences). The full script was run before and after the refactor on the committed CSVs: all six cleaned CSVs are byte-identical and the printed analysis log matches, apart from the removed per-file loading messages. Time-series PNGs differ between two runs of the original script as well, so they were not used for the comparison.

## Structure checks, 2026-10-30

Added `wb_indicators/structure.py`, `scripts/structure_analysis.py` and 17 tests. The tests use planted and random synthetic data: a planted rank-2 structure must beat the column-shuffled null by a wide margin, independent columns must not, held-out cells must be excluded from training, a country-mean baseline must win when years do not vary, and PARAFAC must beat the indicator mean on an exactly low-rank tensor. 25 tests pass in total and `ruff` is clean.

`scripts/structure_analysis.py` ran on all six committed CSVs (215 countries, 2015 to 2024, tensor 215 by 10 by 6 with 78.3% of cells observed) and wrote `outputs/structure_findings.json` and `outputs/structure_checks.png`. The CI verifier checks the JSON for internal consistency. It does not rerun the analysis. The run takes about 15 seconds. Results depend on the scikit-learn and tensorly versions recorded in the JSON.

What the checks found is in the README under key findings. The main points: rank-2 NMF components are stable across seeds and bootstrap resamples but move when GDP and inflation are log-scaled (cosine 0.63 for the second component), NMF error is within about 4% to 7% of column-shuffled data at ranks 2 and 3, and PARAFAC ranks 1 to 5 predict held-out cells worse than a country's own average from other years (0.29 against 0.62 to 1.08 in standardised RMSE).

`world_bank_analysis.py` was not changed, so the earlier NMF and PARAFAC printouts are unchanged. They are now documented as not supported as latent groups.

## Analysis script fixes, 2026-10-30

`scripts/world_bank_analysis.py` printed CRITICAL ERROR when a data file was missing and then exited with status 0, so a CI job or a shell pipeline would have counted the run as a success. It now exits with status 1. Other fixes: the log file is closed and stdout restored (`wb_indicators/runlog.py`), the script no longer changes the working directory and writes under `--output-dir` (default is the repository root, so the usual run is unchanged), `--data-dir` is available, NMF runs with `max_iter=2000` so it converges, `!= True` comparisons became boolean negation, and one unused import and variable were removed. `scripts/` is now linted in CI and `make lint`.

The behaviour was compared before and after on the committed CSVs. All six cleaned CSVs are byte-identical, the printed analysis log is identical including the NMF weights, and both runs wrote 46 figures. Figures were not compared pixel by pixel. 28 tests pass, 3 of them new: the log helper writes to both places and restores stdout even when the block raises, and a missing data file fails the run with status 1.
