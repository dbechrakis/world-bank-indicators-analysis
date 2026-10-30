import unittest
import warnings

import numpy as np
import pandas as pd

from wb_indicators import structure

warnings.filterwarnings("ignore")


def planted_data(rows=120, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    weights = rng.random((rows, 2))
    components = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    return np.clip(weights @ components + rng.random((rows, 6)) * noise, 0, None)


class NmfChecksTests(unittest.TestCase):
    def test_a_planted_structure_beats_the_shuffled_null_by_a_wide_margin(self):
        result = structure.nmf_null_comparison(planted_data(), 2, draws=10)
        self.assertLess(result["real"], 0.5 * result["null_min"])

    def test_independent_columns_do_not_beat_the_shuffled_null(self):
        data = np.random.default_rng(1).random((120, 6))
        result = structure.nmf_null_comparison(data, 2, draws=10)
        self.assertGreater(result["real"], 0.95 * result["null_mean"])

    def test_match_cosines_ignores_component_order_and_scale(self):
        first = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        second = np.array([[0.0, 6.0, 0.0], [2.0, 0.0, 4.0]])
        self.assertTrue(np.allclose(structure.match_cosines(first, second), 1.0))

    def test_match_cosines_is_low_for_different_components(self):
        first = np.eye(3)[:2]
        second = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        self.assertLess(structure.match_cosines(first, second).min(), 0.1)

    def test_shuffle_keeps_each_column_and_breaks_the_pairing(self):
        data = np.column_stack([np.arange(50.0), np.arange(50.0) * 2])
        shuffled = structure.shuffle_columns(data, np.random.default_rng(0))
        self.assertTrue(np.array_equal(np.sort(shuffled[:, 0]), data[:, 0]))
        self.assertFalse(np.array_equal(shuffled[:, 1], shuffled[:, 0] * 2))

    def test_seed_stability_is_high_for_a_clear_structure(self):
        result = structure.nmf_seed_stability(planted_data(), 2, range(5))
        self.assertGreater(result["min"], 0.99)


class PreprocessingTests(unittest.TestCase):
    def table(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(rng.random((30, 6)) * 10 + 1, columns=list("ABCDEF"))
        frame = frame.rename(columns={"D": "GDP", "E": "Inflation"})
        frame.iloc[0, 0] = np.nan
        return frame

    def test_every_preprocessing_returns_finite_nonnegative_data(self):
        for name, prepare in structure.PREPROCESSINGS.items():
            data = prepare(self.table())
            self.assertTrue(np.isfinite(data).all(), name)
            self.assertGreaterEqual(data.min(), 0, name)

    def test_complete_case_drops_countries_with_a_missing_value(self):
        self.assertEqual(len(structure.complete_case_preprocessing(self.table())), 29)

    def test_min_max_scaling_squeezes_a_skewed_column_but_log_scaling_does_not(self):
        table = self.table()
        table["GDP"] = 10 ** np.linspace(0, 6, 30)
        original = structure.original_preprocessing(table)[:, 3]
        logged = structure.log_preprocessing(table)[:, 3]
        self.assertGreater((original < 0.01).mean(), 0.5)
        self.assertLess((logged < 0.01).mean(), 0.1)


class TensorTests(unittest.TestCase):
    def master(self):
        rows = []
        for country in ("A", "B", "C"):
            for year in (2020, 2021):
                for indicator, value in (("X", 1.0), ("Y", 2.0)):
                    if (country, year, indicator) != ("C", 2021, "Y"):
                        rows.append({"Country Name Standardized": country, "Year": year,
                                     "Indicator Short": indicator, "Value": value + len(rows)})
        return pd.DataFrame(rows)

    def test_tensor_has_nan_only_where_the_cell_is_missing(self):
        tensor = structure.tensor_from_master(self.master(), ["X", "Y"])
        self.assertEqual(tensor.shape, (3, 2, 2))
        self.assertEqual(int(np.isnan(tensor).sum()), 1)
        self.assertTrue(np.isnan(tensor[2, 1, 1]))

    def test_zscore_gives_each_indicator_zero_mean_and_unit_spread(self):
        tensor = np.random.default_rng(0).random((10, 4, 3)) * np.array([1, 100, 1e6])
        z = structure.zscore_by_indicator(tensor)
        self.assertTrue(np.allclose(np.nanmean(z, axis=(0, 1)), 0, atol=1e-9))
        self.assertTrue(np.allclose(np.nanstd(z, axis=(0, 1)), 1))

    def test_holdout_removes_only_observed_cells_from_training(self):
        observed = np.random.default_rng(0).random((20, 5, 4)) > 0.2
        train, cells = structure.holdout_cells(observed, 0.1, 0)
        self.assertFalse(train[cells[:, 0], cells[:, 1], cells[:, 2]].any())
        self.assertTrue(observed[cells[:, 0], cells[:, 1], cells[:, 2]].all())
        self.assertEqual(int(observed.sum() - train.sum()), len(cells))

    def test_country_mean_baseline_wins_when_countries_differ_but_years_do_not(self):
        rng = np.random.default_rng(0)
        z = np.repeat(rng.normal(size=(40, 1, 3)), 6, axis=1) + rng.normal(scale=0.05, size=(40, 6, 3))
        train, cells = structure.holdout_cells(np.ones(z.shape, bool), 0.1, 0)
        result = structure.baseline_rmse(z, train, cells)
        self.assertLess(result["country_indicator_mean"], 0.2 * result["indicator_mean"])

    def test_parafac_beats_the_indicator_mean_on_a_low_rank_tensor(self):
        rng = np.random.default_rng(0)
        a, b, c = rng.normal(size=(30, 2)), rng.normal(size=(8, 2)), rng.normal(size=(4, 2))
        z = np.einsum("ir,jr,kr->ijk", a, b, c)
        train, cells = structure.holdout_cells(np.ones(z.shape, bool), 0.1, 0)
        baseline = structure.baseline_rmse(z, train, cells)["indicator_mean"]
        self.assertLess(structure.parafac_heldout_rmse(z, train, cells, 2, 0), 0.5 * baseline)


class PatternTests(unittest.TestCase):
    def test_presence_auc_is_one_for_perfect_separation_and_half_for_noise(self):
        values = np.arange(40.0)
        self.assertEqual(structure.presence_auc(values, values >= 20, draws=200)["auc"], 1.0)
        noise = np.random.default_rng(0).random(2000)
        flags = np.random.default_rng(1).random(2000) > 0.7
        self.assertAlmostEqual(structure.presence_auc(noise, flags, draws=50)["auc"], 0.5, delta=0.05)

    def test_spearman_interval_contains_the_estimate(self):
        x = np.random.default_rng(0).random(60)
        result = structure.spearman_with_ci(x, -x + np.random.default_rng(1).normal(scale=0.1, size=60), draws=300)
        self.assertLess(result["rho"], -0.8)
        self.assertLessEqual(result["ci_low"], result["rho"])
        self.assertGreaterEqual(result["ci_high"], result["rho"])

    def test_inverted_u_is_invisible_to_a_single_correlation(self):
        gdp = np.arange(100.0)
        frame = pd.DataFrame({"GDP": gdp, "R": -((gdp - 50) ** 2)})
        self.assertLess(abs(structure.spearman_with_ci(frame.GDP, frame.R, draws=100)["rho"]), 0.1)
        medians = [q["median"] for q in structure.median_by_quintile(frame, "GDP", "R")]
        self.assertGreater(medians[2], medians[0])
        self.assertGreater(medians[2], medians[4])


if __name__ == "__main__":
    unittest.main()
