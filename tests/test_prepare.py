import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from wb_indicators.outliers import iqr_bounds, iqr_outlier_mask, zscores
from wb_indicators.prepare import (
    country_to_region,
    data_harvesting,
    long_format,
    prepare_master_dataset,
    standardize_country,
)

WORLD_BANK_CSV = """"Data Source","World Development Indicators",

"Last Updated Date","2026-01-01",

"Country Name","Country Code","Indicator Name","Indicator Code","2014","2015","2016",
"Greece","GRC","GDP","NY.GDP","1.0","2.0","",
"Euro area","EMU","GDP","NY.GDP","5.0","6.0","7.0",
"  japan ","JPN","GDP","NY.GDP","3.0","","4.0",
"""


class PrepareTests(unittest.TestCase):
    def harvest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gdp.csv"
            path.write_text(WORLD_BANK_CSV)
            return data_harvesting(path)

    def test_country_lookup(self):
        self.assertEqual(standardize_country("GRC"), "Greece")
        self.assertEqual(standardize_country("EMU"), "Not a country")
        self.assertEqual(country_to_region("GRC"), "Europe")
        self.assertEqual(country_to_region("JPN"), "Asia")
        self.assertEqual(country_to_region("XXX"), "Unknown")

    def test_harvesting_drops_aggregates_and_empty_columns(self):
        df = self.harvest()
        self.assertEqual(sorted(df["Country Code"]), ["GRC", "JPN"])
        self.assertEqual(df.loc[df["Country Code"] == "JPN", "Country Name"].item(), "Japan")
        self.assertFalse(any(str(c).startswith("Unnamed") for c in df.columns))

    def test_long_format_drops_missing_values(self):
        long = long_format(self.harvest())
        self.assertEqual(len(long), 4)
        self.assertTrue(pd.api.types.is_numeric_dtype(long["Year"]))
        self.assertEqual(long["Value"].isna().sum(), 0)

    def test_master_dataset_keeps_latest_window(self):
        df = self.harvest()
        master = prepare_master_dataset([df, df], ["GDP", "GDP2"], window=2)
        self.assertEqual(set(master["Year"]), {2015, 2016})
        self.assertEqual(set(master["Indicator Short"]), {"GDP", "GDP2"})


class OutlierTests(unittest.TestCase):
    def test_constant_series_has_zero_scores(self):
        np.testing.assert_array_equal(zscores([3, 3, 3]), [0, 0, 0])

    def test_zscores_are_standardised(self):
        z = zscores([1, 2, 3, 4, 5])
        self.assertAlmostEqual(z.mean(), 0)
        self.assertAlmostEqual(z.std(), 1)

    def test_iqr_flags_only_extreme_value(self):
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 100], dtype=float)
        lower, upper = iqr_bounds(values)
        self.assertEqual((lower, upper), (-3.0, 13.0))
        self.assertEqual(iqr_outlier_mask(values).tolist(), [False] * 8 + [True])


if __name__ == "__main__":
    unittest.main()
