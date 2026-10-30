"""Test whether the NMF and PARAFAC structure is real, and describe two clear patterns.

Writes outputs/structure_findings.json and outputs/structure_checks.png.
"""

import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import tensorly  # noqa: E402

from wb_indicators import structure  # noqa: E402
from wb_indicators.prepare import data_harvesting, prepare_master_dataset  # noqa: E402

FILES = ["nuclear_production", "renewable_production", "renewable_consumption",
         "gdp", "inflation", "exports"]
NAMES = ["Nuclear", "Renew_Prod", "Renew_Cons", "GDP", "Inflation", "Exports"]
PARAFAC_RANKS = (1, 2, 3, 4, 5)
PARAFAC_SEEDS = (0, 1, 2)


def load_master():
    frames = [data_harvesting(ROOT / "data" / "raw" / f"{name}.csv") for name in FILES]
    return prepare_master_dataset(frames, NAMES)


def nmf_checks(table):
    data = structure.original_preprocessing(table)
    gdp_column = NAMES.index("GDP")
    return {
        "countries": int(len(table)),
        "scaled_gdp_below_0.01_share": float((data[:, gdp_column] < 0.01).mean()),
        "seed_stability_rank2": structure.nmf_seed_stability(data, 2, range(30)),
        "bootstrap_similarity_rank2": structure.bootstrap_component_similarity(data, 2),
        "preprocessing_sensitivity_rank2": structure.preprocessing_sensitivity(table, 2),
        "relative_error_by_rank": structure.error_by_rank(data, range(1, 6)),
        "null_comparison": {str(k): structure.nmf_null_comparison(data, k) for k in (1, 2, 3)},
    }


def parafac_checks(master):
    z = structure.zscore_by_indicator(structure.tensor_from_master(master, NAMES))
    observed = np.isfinite(z)
    train, cells = structure.holdout_cells(observed, 0.1, 0)
    result = {
        "tensor_shape": list(z.shape),
        "observed_share": float(observed.mean()),
        "held_out_cells": int(len(cells)),
        "baselines": structure.baseline_rmse(z, train, cells),
        "parafac": {},
    }
    for rank in PARAFAC_RANKS:
        errors = [structure.parafac_heldout_rmse(z, train, cells, rank, s) for s in PARAFAC_SEEDS]
        result["parafac"][str(rank)] = {"mean": float(np.mean(errors)), "min": float(np.min(errors)),
                                        "max": float(np.max(errors))}
    return result


def pattern_checks(table, master):
    nuclear = table.dropna(subset=["Nuclear", "GDP"])
    present = (nuclear["Nuclear"] > 0).to_numpy()
    producers = nuclear[present]
    region = master.drop_duplicates("Country Name Standardized").set_index("Country Name Standardized")["Region"]
    top = producers["Nuclear"].sort_values(ascending=False).head(6)
    return {
        "nuclear": {
            "countries_with_data": int(len(nuclear)),
            "countries_with_nuclear": int(present.sum()),
            "by_region": {k: int(v) for k, v in region.loc[producers.index].value_counts().items()},
            "gdp_predicts_presence": structure.presence_auc(nuclear["GDP"], present),
            "gdp_vs_share_among_producers": structure.spearman_with_ci(producers["GDP"], producers["Nuclear"]),
            "highest_shares": {k: float(v) for k, v in top.items()},
        },
        "renewable_consumption": {
            "gdp_overall": structure.spearman_with_ci(*table[["GDP", "Renew_Cons"]].dropna().T.to_numpy()),
            "median_by_gdp_quintile": structure.median_by_quintile(table, "GDP", "Renew_Cons"),
        },
    }


def draw(findings, path):
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    ranks = [1, 2, 3]
    real = [findings["nmf"]["null_comparison"][str(k)]["real"] for k in ranks]
    null_low = [findings["nmf"]["null_comparison"][str(k)]["null_min"] for k in ranks]
    null_high = [findings["nmf"]["null_comparison"][str(k)]["null_max"] for k in ranks]
    left.fill_between(ranks, null_low, null_high, alpha=0.3, label="Column-shuffled data, min to max of 40")
    left.plot(ranks, real, marker="o", label="Real data")
    left.set_xticks(ranks)
    left.set_xlabel("NMF rank")
    left.set_ylabel("Relative reconstruction error")
    left.set_title("NMF barely beats shuffled columns")
    left.legend()
    labels = ["Indicator\nmean"] + [f"PARAFAC\nrank {k}" for k in PARAFAC_RANKS[:4]] + ["Country's own\nmean"]
    values = ([findings["parafac"]["baselines"]["indicator_mean"]]
              + [findings["parafac"]["parafac"][str(k)]["mean"] for k in PARAFAC_RANKS[:4]]
              + [findings["parafac"]["baselines"]["country_indicator_mean"]])
    right.bar(range(len(values)), values)
    right.set_xticks(range(len(values)))
    right.set_xticklabels(labels, fontsize=8)
    right.set_ylabel("Held-out RMSE, standardised units")
    right.set_title("A country's own average beats the tensor model")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main():
    master = load_master()
    table = structure.country_table(master, NAMES)
    findings = {
        "versions": {"scikit-learn": sklearn.__version__, "tensorly": tensorly.__version__,
                     "numpy": np.__version__},
        "window_years": [int(master["Year"].min()), int(master["Year"].max())],
        "nmf": nmf_checks(table),
        "parafac": parafac_checks(master),
        "patterns": pattern_checks(table, master),
    }
    out = ROOT / "outputs"
    (out / "structure_findings.json").write_text(json.dumps(findings, indent=2) + "\n")
    draw(findings, out / "structure_checks.png")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
