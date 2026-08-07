import pandas as pd
import numpy as np
import pycountry
import pycountry_convert as pc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Advanced libraries we needed for the project requirements
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from tensorly.decomposition import parafac
from matplotlib.colors import SymLogNorm

# Making sure we have a place to save the output graphs
if not os.path.exists('graphs'):
    os.makedirs('graphs')

class DualLogger:
    """
    We added this helper class so we could see the output in the terminal
    BUT also save it to a text file
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout so everything prints to the log file too
sys.stdout = DualLogger("project_output.txt")


# ==========================================
# 1. DATA HARVESTING & CLEANING
# ==========================================

def data_harvesting(file):
    print(f"\n--- Processing File: {file} ---")
    
    # World Bank data always has 4 rows of metadata at the top, so we skip them
    df = pd.read_csv(file, skiprows=4)
    
    # Dropping columns that are completely empty
    df = df.dropna(axis=1, how='all')
        
    def country_standardized(code):
        # Cleaning up the names in case of whitespace
        df['Country Name'] = df['Country Name'].str.strip().str.title()
        try:
            # We used pycountry to turn the 3-letter code into the official name
            # If the country code doesn't correspond to any official name, it gets flagged
            country = pycountry.countries.get(alpha_3=code)
            return country.name if country else "Not a country"
        except:
            return "Not a country"
            
    df['Country Name Standardized'] = df['Country Code'].apply(country_standardized)
    
    # We filter out "Not a country" rows because we want country-level data
    df = df[df['Country Name Standardized'] != 'Not a country']
    
    def country_to_region(code):
        try:
            country = pycountry.countries.get(alpha_3=code)
            if country is None: return "Unknown"
            
            # Mapping countries to Continents
            region_code = pc.country_alpha2_to_continent_code(country.alpha_2)
            
            continent_map = {
                "AF": "Africa", "AS": "Asia", "EU": "Europe",
                "NA": "North America", "SA": "South America",
                "OC": "Oceania", "AN": "Antarctica"
            }
            return continent_map.get(region_code, "Unknown")
        except:
            # Handling edge cases when not in the library
            return "Unknown"
            
    df['Region'] = df['Country Code'].apply(country_to_region)
    print(f"Loaded and standardized {file} successfully.")
    return df

def long_format(df):

    # Transforming from Wide format (Years as columns) to Long format.
    
    id_vars = [
        'Country Code',
        'Country Name Standardized',
        'Region',
        'Indicator Name',
        'Indicator Code'
    ]
    
    # Grab all the remaining columns (which are the Years)
    year_cols = [c for c in df.columns if c not in id_vars]

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name='Year',
        value_name='Value'
    )

    df_long = df_long.dropna(subset=['Value'])
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')

    return df_long

def export_indicator_csv(df, unit, output_filename):
    # We saved the cleaned data to CSVs
    if not os.path.exists('cleaned_data'):
        os.makedirs('cleaned_data')
        
    df_long = long_format(df)
    df_long = df_long.rename(columns={'Country Name Standardized': 'Country Name'})
    df_long['Unit'] = unit

    # Reordering columns
    cols = ['Country Code', 'Country Name', 'Region', 'Indicator Code', 'Indicator Name', 'Unit', 'Value', 'Year']
    df_long = df_long[cols]

    output_path = os.path.join('cleaned_data', output_filename)
    df_long.to_csv(output_path, index=False)
    print(f"Exported clean data to: {output_filename}")


# ==========================================
# 2. OUTLIER DETECTION
# ==========================================

def outliers_last10(df, indicator_label): 
# We chose the last 10 years for our analysis 
    df_long = long_format(df) 
    current_year = df_long['Year'].max() 
    df_long = df_long[df_long['Year'] >= (current_year - 9)].copy() 
    print(f"\nAnalyzing Outliers (Last 10 Years, By Region): {indicator_label}\n") 
    df_long['Z Score'] = pd.Series(np.nan, index=df_long.index, dtype='float64')
    df_long['Outlier'] = False 
    
    # We calculate Z-scores per region
    for region, region_data in df_long.groupby('Region'): 
        values = region_data['Value'].to_numpy() 
        mean = np.nanmean(values) 
        std = np.nanstd(values) 
        if std == 0: 
            z = np.zeros(len(values)) 
        else: 
            z = (values - mean) / std 
        # Assign Z-scores back 
        df_long.loc[region_data.index, 'Z Score'] = z.astype('float64') 
        # Standard IQR method for finding the thresholds, per region 
        Q1 = np.nanpercentile(z, 25) 
        Q3 = np.nanpercentile(z, 75) 
        IQR = Q3 - Q1 
        lower = Q1 - 1.5 * IQR 
        upper = Q3 + 1.5 * IQR 
        outliers = (z < lower) | (z > upper) 
        df_long.loc[region_data.index, 'Outlier'] = outliers 
    
    total_outliers = df_long['Outlier'].sum() 
    percent_outliers = df_long['Outlier'].mean() * 100 
    print(f"Total outliers found: {total_outliers} ({percent_outliers:.2f}%)")
        
    # Boxplot BEFORE removing outliers
    plt.figure(figsize=(6, 8))
    df_long['Z Score'].plot(kind='box')
    plt.title(f'{indicator_label} - Before Removing Outliers')
    plt.ylabel('Z Score')
    plt.tight_layout()
    plt.savefig(f"graphs/Boxplot_Before_{indicator_label.replace(' ', '_')}.png")
    plt.close()
    
    # Remove outliers
    removed_count = df_long['Outlier'].sum()
    df_long = df_long[df_long['Outlier'] != True].copy()

    # Boxplot AFTER removing outliers, include number removed in title
    plt.figure(figsize=(6, 8))
    df_long['Z Score'].plot(kind='box', color='green')
    plt.title(f'{indicator_label} - After Removing Outliers\nRemoved: {removed_count} values, Percent of Total: {percent_outliers:.2f}%')
    plt.ylabel('Z Score')
    plt.tight_layout()    
    plt.savefig(f"graphs/Boxplot_After_{indicator_label.replace(' ', '_')}.png")
    plt.close()
    
    return df_long

def europe_decade_outliers(df, indicator_label):
    # For the specific requirement to look at a region in decade intervals
    # our choice was Europe
    df_long = long_format(df)
    region_df = df_long[df_long['Region'] == "Europe"].copy()
    
    # We built the time bins dynamically based on the data range
    min_year, max_year = int(region_df['Year'].min()), int(region_df['Year'].max())
    intervals = []
    start = min_year
    while start <= max_year:
        end = min(start + 9, max_year)
        intervals.append((start, end))
        start = end + 1
        
    def assign_interval(year):
        for s, e in intervals:
            if s <= year <= e: return f"{s}-{e}"
        return None

    region_df["Interval"] = region_df["Year"].apply(assign_interval)
    
    # Calculating Z-scores inside each specific time bucket
    for interval, interval_df in region_df.groupby('Interval'):
        values = interval_df['Value'].to_numpy()
        mean = np.nanmean(values)
        std = np.nanstd(values)
        if std == 0:
            z = np.zeros(len(values))
        else:
            z = (values - mean) / std
        region_df.loc[interval_df.index, 'Z Score'] = z

    Q1 = region_df['Z Score'].quantile(0.25)
    Q3 = region_df['Z Score'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    region_df['Outlier'] = (region_df['Z Score'] < lower) | (region_df['Z Score'] > upper)
    europe_removed_count = region_df['Outlier'].sum()
    percent_removed = europe_removed_count / len(region_df) * 100

    print(f"Europe Decade Thresholds -> L: {lower:.3f}, U: {upper:.3f}")
    print(f"Europe Outliers Found: {europe_removed_count}")
    
    
    # Boxplot BEFORE removing outliers, with highlighted outliers
    plt.figure(figsize=(6, 8))
    ax = region_df['Z Score'].dropna().plot(kind='box')
    
    ax.axhline(lower, linestyle='--', linewidth=2, color='red')
    ax.axhline(upper, linestyle='--', linewidth=2, color='red')
    
    # Add numeric labels for thresholds
    ax.text(
        1.05, lower, f"Lower = {lower:.2f}",
        transform=ax.get_yaxis_transform(),
        color='red', fontsize=9
    )
    ax.text(
        1.05, upper, f"Upper = {upper:.2f}",
        transform=ax.get_yaxis_transform(),
        color='red', fontsize=9
    )
    
    plt.title(f'Europe Decades - Before Outlier Removal\n{indicator_label}')
    plt.ylabel('Z Score')
    plt.tight_layout()
    plt.savefig(f"graphs/Boxplot_Europe_Before_{indicator_label.replace(' ', '_')}.png")
    plt.close()

    # Boxplot AFTER removing outliers, include number removed in title
    region_df = region_df[region_df['Outlier'] != True].copy()

    plt.figure(figsize=(6, 8))
    region_df['Z Score'].dropna().plot(kind='box', color='green')
    plt.title(
        f'Europe Decades - After Outlier Removal\n'
        f'{indicator_label}\n'
        f'Removed: {europe_removed_count} values ({percent_removed:.2f}%)'
    )
    plt.ylabel('Z Score')
    plt.tight_layout()
    plt.savefig(f"graphs/Boxplot_Europe_After_{indicator_label.replace(' ', '_')}.png")
    plt.close()
    plt.close()
    
    return region_df


# ==========================================
# 3. MASTER DATA PREP
# ==========================================

def prepare_master_dataset(dfs, indicator_names):
    # We combine all the individual indicator dataframes into one huge master file
    master_list = []
    for df, name in zip(dfs, indicator_names):
        df_long = long_format(df)
        df_long['Indicator Short'] = name
        master_list.append(df_long)
    
    full_df = pd.concat(master_list, ignore_index=True)
    
    # We filter to the last 10 years again
    max_year = full_df['Year'].max()
    full_df = full_df[full_df['Year'] >= (max_year - 9)]
    return full_df


# ==========================================
# 4. VISUALIZATION
# ==========================================

def generate_time_series(master_df):
    # Creating time series plots for every region to spot trends
    print("\n=== PLOTTING TIME SERIES ===")
    regions = master_df['Region'].unique()
    indicators = master_df['Indicator Short'].unique()
    
    for region in regions:
        region_data = master_df[master_df['Region'] == region]
        if region_data.empty: continue
        
        # We used a 3x2 grid so we can see all 6 indicators at once
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Time Series Trends (Last 10 Years) - {region}', fontsize=16)
        axes = axes.flatten()
        
        for i, indicator in enumerate(indicators):
            ind_data = region_data[region_data['Indicator Short'] == indicator]
            if ind_data.empty: continue
            
            # We decided to use Seaborn here because it handles the confidence interval shading automatically
            sns.lineplot(data=ind_data, x='Year', y='Value', ax=axes[i], marker='o')

            axes[i].set_title(indicator)
            axes[i].set_xlabel('Year')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"graphs/TimeSeries_{region}.png")
        plt.close()
        print(f"Graph saved for: {region}")

def analyze_relationships(master_df):
    # Generating Heatmaps for both Correlation and Covariance
    print("\n=== GENERATING HEATMAPS ===")
    regions = master_df['Region'].unique()
    
    for region in regions:
        region_data = master_df[master_df['Region'] == region]
        if region_data.empty: continue

        # We need to pivot so indicators become columns
        pivot_df = region_data.pivot_table(
            index=['Country Name Standardized', 'Year'], 
            columns='Indicator Short', 
            values='Value'
        )
        
        # 1. Correlation Matrix
        plt.figure(figsize=(10, 8))
        corr_matrix = pivot_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Correlation Matrix - {region}')
        plt.tight_layout()
        plt.savefig(f"graphs/Heatmap_Correlation_{region}.png")
        plt.close()

        # 2. Covariance Matrix
        cov_matrix = pivot_df.cov()
        print(f"\nCovariance for {region}:")
        print(cov_matrix)
        
        # We had to use a Log Scale (SymLogNorm) here because GDP is in Trillions while other data is small percentages
        plt.figure(figsize=(10, 8))
        try:
            sns.heatmap(cov_matrix, annot=False, cmap='viridis', 
                        norm=SymLogNorm(linthresh=1.0, vmin=cov_matrix.min().min(), vmax=cov_matrix.max().max()))
            plt.title(f'Covariance Matrix (Log Scale) - {region}')
        except ValueError:
            # Fallback if the matrix is problematic (e.g. all zeros)
            sns.heatmap(cov_matrix, annot=False, cmap='viridis')
            plt.title(f'Covariance Matrix - {region}')
            
        plt.tight_layout()
        plt.savefig(f"graphs/Heatmap_Covariance_{region}.png")
        plt.close()


# ==========================================
# 5. ADVANCED MODELS
# ==========================================

def perform_nmf(master_df):
    # We apply NMF here to try and find natural groupings in the data
    print("\n=== NMF ANALYSIS ===")
    
    # Pivot to Country x Indicator
    nmf_data = master_df.pivot_table(
        index='Country Name Standardized', 
        columns='Indicator Short', 
        values='Value', aggfunc='mean'
    ).fillna(0)
    
    # We must scale the data to 0-1 because NMF breaks with negative numbers
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(nmf_data)
    
    model = NMF(n_components=2, init='random', random_state=42)
    W = model.fit_transform(data_scaled)
    H = model.components_
    
    indicators = nmf_data.columns
    for i, topic in enumerate(H):
        print(f"\nGroup {i+1} Weights:")
        # Sorting weights so we can see which indicators belong to which group
        top_indices = topic.argsort()[::-1]
        for idx in top_indices:
            print(f"  {indicators[idx]}: {topic[idx]:.4f}")

def perform_tensor_decomp(master_df):
    # This is the Extra Work part: Tensor decomposition (3D: Country x Year x Indicator)
    print("\n=== TENSOR DECOMPOSITION ===")
    
    # 1. Pivot but keep Year in the index
    tensor_df = master_df.pivot_table(
        index=['Country Name Standardized', 'Year'],
        columns='Indicator Short',
        values='Value'
    ).dropna()
    
    countries = tensor_df.index.get_level_values(0).unique()
    years = tensor_df.index.get_level_values(1).unique()
    indicators = tensor_df.columns
    
    # 2. Building the 3D numpy array
    tensor_shape = (len(countries), len(years), len(indicators))
    tensor = np.zeros(tensor_shape)
    
    for c, country in enumerate(countries):
        for y, year in enumerate(years):
            try:
                val = tensor_df.loc[(country, year)].values
                tensor[c, y, :] = val
            except KeyError:
                pass

    # 3. Z-score normalization per indicator slice
    # We realized this was necessary so that GDP (trillions) doesn't overpower everything else
    for i in range(len(indicators)):
        indicator_slice = tensor[:, :, i]
        mean = np.mean(indicator_slice)
        std = np.std(indicator_slice)
        if std > 0:
            tensor[:, :, i] = (indicator_slice - mean) / std
            
    # 4. Running PARAFAC decomposition
    rank = 3
    weights, factors = parafac(tensor, rank=rank)
    
    # The 3rd factor corresponds to the Indicators
    indicator_factors = factors[2]
    
    print("\nTensor Patterns Found:")
    for i in range(rank):
        print(f"\nPattern {i+1}:")
        weighted = zip(indicators, indicator_factors[:, i])
        sorted_w = sorted(weighted, key=lambda x: abs(x[1]), reverse=True)
        for ind, w in sorted_w:
            print(f"  {ind}: {w:.4f}")

    # Plotting the latent patterns
    plt.figure(figsize=(10, 5))
    plt.plot(indicators, indicator_factors)
    plt.legend([f'Pattern {i+1}' for i in range(rank)])
    plt.title("Tensor Patterns: Indicator Groups (Normalized)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("graphs/Tensor_Patterns.png")
    plt.close()


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    print("Starting the analysis pipeline...\n")
    
    try:
        # Loading our 6 datasets
        df1 = data_harvesting('nuclear_production.csv')
        export_indicator_csv(df1, '%', 'nuclear_production_cleaned.csv')
        
        df2 = data_harvesting('renewable_production.csv')
        export_indicator_csv(df2, '%', 'renewable_production_cleaned.csv')
        
        df3 = data_harvesting('renewable_consumption.csv')
        export_indicator_csv(df3, '%', 'renewable_consumption_cleaned.csv')
        
        df4 = data_harvesting('gdp.csv')
        export_indicator_csv(df4, '$', 'gdp_cleaned.csv')
        
        df5 = data_harvesting('inflation.csv')
        export_indicator_csv(df5, '%', 'inflation_cleaned.csv')
        
        df6 = data_harvesting('exports.csv')
        export_indicator_csv(df6, '%', 'exports_cleaned.csv')
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit()

    # Outlier checks
    print("\n--- Running Outlier Checks ---")
    outliers_last10(df1, "Nuclear Production")
    europe_decade_outliers(df1, "Nuclear Production")
    
    outliers_last10(df2, "Renewable Production")
    europe_decade_outliers(df2, "Renewable Production")
    
    outliers_last10(df3, "Renewable Consumption")
    europe_decade_outliers(df3, "Renewable Consumption")
    
    outliers_last10(df4, "GDP")
    europe_decade_outliers(df4, "GDP")
    
    outliers_last10(df5, "Inflation")
    europe_decade_outliers(df5, "Inflation")
    
    outliers_last10(df6, "Exports")
    europe_decade_outliers(df6, "Exports")

    # Merging for advanced models
    dfs = [df1, df2, df3, df4, df5, df6]
    names = ["Nuclear", "Renew_Prod", "Renew_Cons", "GDP", "Inflation", "Exports"]
    master_df = prepare_master_dataset(dfs, names)
    
    # Generating Plots
    generate_time_series(master_df) 
    analyze_relationships(master_df)
    
    # Running Models
    perform_nmf(master_df)
    perform_tensor_decomp(master_df)

    print("\nDone. All results saved to the graphs/ folder.")
