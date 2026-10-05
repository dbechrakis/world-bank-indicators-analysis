# Validation record

Review date: 2026-09-05 (UTC).

Executed both full analysis and SQL loading against all six committed CSVs; checked row grain and foreign keys.

Historical records are not labelled as freshly reproduced results.

## Refactor check — 2026-10-05

Moved data preparation and outlier statistics out of `scripts/world_bank_analysis.py` into the `wb_indicators` package and added 7 unit tests (country lookup, aggregate removal, long format, latest-year window, z-scores, IQR fences). The full script was run before and after the refactor on the committed CSVs: all six cleaned CSVs are byte-identical and the printed analysis log matches, apart from the removed per-file loading messages. Time-series PNGs differ between two runs of the original script as well, so they were not used for the comparison.
