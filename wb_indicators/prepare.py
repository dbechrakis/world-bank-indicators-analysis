"""Loading, standardising and reshaping World Bank indicator files."""

import pandas as pd
import pycountry
import pycountry_convert as pc

CONTINENTS = {
    "AF": "Africa", "AS": "Asia", "EU": "Europe",
    "NA": "North America", "SA": "South America",
    "OC": "Oceania", "AN": "Antarctica",
}
ID_VARS = [
    "Country Code",
    "Country Name Standardized",
    "Region",
    "Indicator Name",
    "Indicator Code",
]


def standardize_country(code):
    """Official country name for an ISO alpha-3 code, or "Not a country" (e.g. aggregates)."""
    try:
        country = pycountry.countries.get(alpha_3=code)
    except (KeyError, LookupError):
        return "Not a country"
    return country.name if country else "Not a country"


def country_to_region(code):
    """Continent name for an ISO alpha-3 code, or "Unknown"."""
    try:
        country = pycountry.countries.get(alpha_3=code)
        if country is None:
            return "Unknown"
        return CONTINENTS.get(pc.country_alpha2_to_continent_code(country.alpha_2), "Unknown")
    except (KeyError, LookupError):
        return "Unknown"


def data_harvesting(file):
    """Read one World Bank CSV (4 metadata rows), keep real countries, add Region."""
    df = pd.read_csv(file, skiprows=4)
    df = df.dropna(axis=1, how="all")
    df["Country Name"] = df["Country Name"].str.strip().str.title()
    df["Country Name Standardized"] = df["Country Code"].apply(standardize_country)
    df = df[df["Country Name Standardized"] != "Not a country"]
    df["Region"] = df["Country Code"].apply(country_to_region)
    return df


def long_format(df):
    """Wide (years as columns) to long format, dropping missing values."""
    year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
    df_long = df.melt(id_vars=ID_VARS, value_vars=year_cols, var_name="Year", value_name="Value")
    df_long = df_long.dropna(subset=["Value"])
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    return df_long


def prepare_master_dataset(dfs, indicator_names, window=10):
    """Stack indicators into one long frame limited to the latest `window` years."""
    master_list = []
    for df, name in zip(dfs, indicator_names):
        df_long = long_format(df)
        df_long["Indicator Short"] = name
        master_list.append(df_long)
    full_df = pd.concat(master_list, ignore_index=True)
    max_year = full_df["Year"].max()
    return full_df[full_df["Year"] >= (max_year - (window - 1))]
