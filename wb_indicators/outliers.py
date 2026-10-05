"""Outlier statistics (no plotting, no I/O) used by the analysis script."""

import numpy as np


def zscores(values):
    """Population z-scores; all zeros when the series is constant."""
    values = np.asarray(values, dtype=float)
    std = np.nanstd(values)
    if std == 0:
        return np.zeros(len(values))
    return (values - np.nanmean(values)) / std


def iqr_bounds(scores, multiplier=1.5):
    """Tukey fences (lower, upper) for an array of scores."""
    q1, q3 = np.nanpercentile(scores, [25, 75])
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def iqr_outlier_mask(scores, multiplier=1.5):
    """Boolean mask of scores outside the Tukey fences."""
    scores = np.asarray(scores, dtype=float)
    lower, upper = iqr_bounds(scores, multiplier)
    return (scores < lower) | (scores > upper)
