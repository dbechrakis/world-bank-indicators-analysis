# World Bank Indicators Analysis

A data-science project combining World Bank indicator data, Python analytics, SQLite, statistical analysis, visualization, and unsupervised learning.

## Business / analytical focus

The project studies six indicators across countries and regions:

- GDP
- Inflation
- Exports (% of GDP)
- Nuclear electricity production
- Renewable electricity production
- Renewable energy consumption

The analysis covers data cleaning and standardization, regional aggregation, time-series analysis, correlation and covariance, outlier detection, and advanced pattern-discovery techniques.

## Analytical workflow

1. Load World Bank indicator CSV files.
2. Standardize country names and map countries to continents.
3. Reshape the World Bank wide-format data into a long analytical format.
4. Export cleaned indicator datasets.
5. Detect outliers using regional Z-scores and IQR thresholds.
6. Compare European observations across decade intervals.
7. Build regional time-series visualizations.
8. Analyze relationships between indicators using correlation and covariance matrices.
9. Apply NMF to identify latent indicator groupings.
10. Apply PARAFAC tensor decomposition to explore country × year × indicator patterns.
11. Load the analytical data into SQLite and run SQL-based regional and country queries.

## Repository structure

```text
world-bank-indicators-analysis/
├── README.md
├── scripts/
│   └── world_bank_analysis.py
├── sql/
│   └── build_database.py
└── data/
    └── raw/
        ├── nuclear_production.csv
        ├── renewable_production.csv
        ├── renewable_consumption.csv
        ├── gdp.csv
        ├── inflation.csv
        └── exports.csv
```

## Outputs

The analysis script generates graphs, cleaned datasets, and a project log. These generated artifacts should remain separate from the source data and analysis code.

## Technical stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, TensorLy, pycountry, pycountry-convert, SQLite.

## Portfolio note

This repository presents the analytical workflow as a portfolio project. The emphasis is on reproducibility, clear project structure, data preparation, statistical reasoning, and communicating analytical findings rather than presenting observational relationships as causal effects.
