"""Checks on whether the multivariate structure in the indicator panel is real.

NMF and PARAFAC always return components. These functions ask whether the components are
stable under resampling and preprocessing, whether they beat a null built from the same
marginal distributions, and whether the tensor model predicts held-out cells better than
a trivial baseline.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler


def country_table(master, names):
    """Country by indicator table of ten-year means, columns in the order of names."""
    table = master.pivot_table(
        index="Country Name Standardized", columns="Indicator Short", values="Value", aggfunc="mean"
    )
    return table[names]


def original_preprocessing(table):
    """Median imputation then min-max scaling, as the first version of the script did."""
    filled = table.fillna(table.median())
    return MinMaxScaler().fit_transform(filled)


def log_preprocessing(table):
    """Log10 GDP and log1p inflation before median imputation and min-max scaling."""
    shaped = table.copy()
    shaped["GDP"] = np.log10(shaped["GDP"])
    shaped["Inflation"] = np.log1p(shaped["Inflation"].clip(lower=0))
    return MinMaxScaler().fit_transform(shaped.fillna(shaped.median()))


def rank_preprocessing(table):
    """Percentile ranks within each indicator, with missing values at the median rank."""
    return table.rank(pct=True).fillna(0.5).to_numpy()


def complete_case_preprocessing(table):
    """Log GDP, keep only countries with all indicators, min-max scale. No imputation."""
    complete = table.dropna().copy()
    complete["GDP"] = np.log10(complete["GDP"])
    return MinMaxScaler().fit_transform(complete)


PREPROCESSINGS = {
    "original": original_preprocessing,
    "log": log_preprocessing,
    "rank": rank_preprocessing,
    "complete_case": complete_case_preprocessing,
}


def nmf_fit(data, components, seed=42):
    """Component matrix and relative reconstruction error of one NMF fit."""
    model = NMF(n_components=components, init="random", random_state=seed, max_iter=2000)
    model.fit(data)
    return model.components_, model.reconstruction_err_ / np.linalg.norm(data)


def match_cosines(first, second):
    """Cosine similarity of each component in first with its best one-to-one match in second."""
    a = first / np.linalg.norm(first, axis=1, keepdims=True)
    b = second / np.linalg.norm(second, axis=1, keepdims=True)
    similarity = a @ b.T
    rows, cols = linear_sum_assignment(-similarity)
    return similarity[rows, cols]


def shuffle_columns(data, rng):
    """Permute every column independently. Keeps each marginal and destroys joint structure."""
    return np.column_stack([rng.permutation(data[:, j]) for j in range(data.shape[1])])


def nmf_null_comparison(data, components, draws=40, seed=0):
    """Relative error of the real fit next to fits on column-shuffled copies of the data."""
    rng = np.random.default_rng(seed)
    real = nmf_fit(data, components)[1]
    null = np.array([nmf_fit(shuffle_columns(data, rng), components)[1] for _ in range(draws)])
    return {"real": float(real), "null_mean": float(null.mean()),
            "null_min": float(null.min()), "null_max": float(null.max()), "draws": draws}


def nmf_seed_stability(data, components, seeds):
    """Worst component match between the seed-42 fit and fits from other random starts."""
    reference = nmf_fit(data, components, 42)[0]
    worst = [match_cosines(reference, nmf_fit(data, components, s)[0]).min() for s in seeds]
    return {"min": float(np.min(worst)), "median": float(np.median(worst))}


def bootstrap_component_similarity(data, components, draws=200, seed=0):
    """Worst component match between the full-data fit and fits on resampled countries."""
    rng = np.random.default_rng(seed)
    reference = nmf_fit(data, components)[0]
    worst = []
    for _ in range(draws):
        resample = data[rng.integers(0, len(data), len(data))]
        worst.append(match_cosines(reference, nmf_fit(resample, components)[0]).min())
    low, median, high = np.quantile(worst, [0.05, 0.5, 0.95])
    return {"q05": float(low), "median": float(median), "q95": float(high), "draws": draws}


def preprocessing_sensitivity(table, components=2):
    """Per-component match between the original-preprocessing fit and each alternative."""
    reference = nmf_fit(original_preprocessing(table), components)[0]
    return {
        name: [float(x) for x in match_cosines(reference, nmf_fit(prepare(table), components)[0])]
        for name, prepare in PREPROCESSINGS.items() if name != "original"
    }


def error_by_rank(data, ranks, seeds=(0, 1, 2, 3, 4)):
    """Mean relative reconstruction error per rank over several random starts."""
    return {int(k): float(np.mean([nmf_fit(data, k, s)[1] for s in seeds])) for k in ranks}


def tensor_from_master(master, names):
    """Country by year by indicator array with NaN for missing cells."""
    table = master.pivot_table(
        index=["Country Name Standardized", "Year"], columns="Indicator Short", values="Value"
    )[names]
    countries = table.index.get_level_values(0).unique()
    years = table.index.get_level_values(1).unique()
    tensor = np.full((len(countries), len(years), len(names)), np.nan)
    rows = countries.get_indexer(table.index.get_level_values(0))
    cols = years.get_indexer(table.index.get_level_values(1))
    tensor[rows, cols, :] = table.to_numpy()
    return tensor


def zscore_by_indicator(tensor):
    """Standardise each indicator slice over its observed cells."""
    scaled = tensor.copy()
    for j in range(tensor.shape[2]):
        slice_ = tensor[:, :, j]
        std = np.nanstd(slice_)
        scaled[:, :, j] = (slice_ - np.nanmean(slice_)) / std if std > 0 else 0.0
    return scaled


def holdout_cells(observed, fraction=0.1, seed=0):
    """Training mask with a random share of observed cells removed, and those cells' indices."""
    rng = np.random.default_rng(seed)
    cells = np.argwhere(observed)
    held = cells[rng.random(len(cells)) < fraction]
    train = observed.copy()
    train[held[:, 0], held[:, 1], held[:, 2]] = False
    return train, held


def _rmse(truth, prediction, cells):
    index = (cells[:, 0], cells[:, 1], cells[:, 2])
    return float(np.sqrt(np.mean((truth[index] - prediction[index]) ** 2)))


def baseline_rmse(z, train, cells):
    """Held-out error of two trivial predictors, in standardised units."""
    masked = np.where(train, z, np.nan)
    indicator_mean = np.nanmean(masked, axis=(0, 1))
    country_mean = np.nanmean(masked, axis=1)
    country_mean = np.where(np.isfinite(country_mean), country_mean, indicator_mean[None, :])
    indicator_prediction = np.broadcast_to(indicator_mean, z.shape)
    country_prediction = np.broadcast_to(country_mean[:, None, :], z.shape)
    return {"indicator_mean": _rmse(z, indicator_prediction, cells),
            "country_indicator_mean": _rmse(z, country_prediction, cells)}


def parafac_heldout_rmse(z, train, cells, rank, seed):
    """Held-out error of a masked PARAFAC fit of the given rank."""
    from tensorly.decomposition import parafac

    weights, factors = parafac(np.nan_to_num(z, nan=0.0), rank=rank, mask=train,
                               init="random", random_state=seed, n_iter_max=500)
    reconstruction = np.einsum("r,ir,jr,kr->ijk", weights, *factors)
    return _rmse(z, reconstruction, cells)


def presence_auc(values, present, draws=2000, seed=0):
    """Probability that a random country with the trait outranks one without, with a bootstrap CI."""
    values = np.asarray(values, dtype=float)
    present = np.asarray(present, dtype=bool)

    def auc(v, p):
        return mannwhitneyu(v[p], v[~p]).statistic / (p.sum() * (~p).sum())

    rng = np.random.default_rng(seed)
    draws_ = []
    for _ in range(draws):
        pick = rng.integers(0, len(values), len(values))
        if present[pick].sum() > 1 and (~present[pick]).sum() > 1:
            draws_.append(auc(values[pick], present[pick]))
    low, high = np.quantile(draws_, [0.025, 0.975])
    return {"auc": float(auc(values, present)), "ci_low": float(low), "ci_high": float(high)}


def spearman_with_ci(x, y, draws=2000, seed=0):
    """Spearman correlation with a percentile bootstrap interval over pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        pick = rng.integers(0, len(x), len(x))
        rho = spearmanr(x[pick], y[pick]).statistic
        if np.isfinite(rho):
            boot.append(rho)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {"rho": float(spearmanr(x, y).statistic), "ci_low": float(low), "ci_high": float(high),
            "n": int(len(x))}


def median_by_quintile(frame, by, value):
    """Median of value within quintiles of by, with the number of countries in each."""
    both = frame[[by, value]].dropna()
    both["quintile"] = pd.qcut(both[by], 5, labels=False)
    grouped = both.groupby("quintile")[value].agg(["count", "median"])
    return [{"quintile": int(q), "countries": int(r["count"]), "median": float(r["median"])}
            for q, r in grouped.iterrows()]
