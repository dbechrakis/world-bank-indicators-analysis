import sqlite3
import pandas as pd
import sys
import os
import pycountry
import pycountry_convert as pc

# --- 1. SETUP PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))

def get_file_path(filename):
    return os.path.join(script_dir, filename)

# --- 2. DATA PROCESSING LOGIC ---

def data_harvesting(file):
    if not os.path.exists(file):
        print(f"Error: Could not find {file}")
        return pd.DataFrame()

    print(f"Loading {file}...")
    df = pd.read_csv(file, skiprows=4)
    df = df.dropna(axis=1, how='all')
    
    if 'Indicator Code' in df.columns:
        df = df.drop(columns=['Indicator Code'])
        
    def country_standardized(code):
        try:
            country = pycountry.countries.get(alpha_3=code)
            return country.name if country else "Region"
        except:
            return "Region"
            
    df['Country Name Standardized'] = df['Country Code'].apply(country_standardized)
    df = df[df['Country Name Standardized'] != 'Region']
    
    def country_to_region(code):
        try:
            country = pycountry.countries.get(alpha_3=code)
            if country is None: return "Unknown"
            region_code = pc.country_alpha2_to_continent_code(country.alpha_2)
            continent_map = {
                "AF": "Africa", "AS": "Asia", "EU": "Europe",
                "NA": "North America", "SA": "South America",
                "OC": "Oceania", "AN": "Antarctica"
            }
            return continent_map.get(region_code, "Unknown")
        except:
            return "Unknown"
            
    df['Region'] = df['Country Code'].apply(country_to_region)
    return df

def long_format_corrected(df):
    if df.empty: return df
    
    # THE FIX: We explicitly include 'Country Code' in id_vars so it is kept
    id_vars = ['Country Name', 'Country Code', 'Country Name Standardized', 'Region', 'Indicator Name']
    year_cols = [c for c in df.columns if c not in id_vars]
    
    df_long = df.melt(
        id_vars=['Country Name', 'Country Code', 'Country Name Standardized', 'Region', 'Indicator Name'],
        value_vars=year_cols,
        var_name='Year',
        value_name='Value'
    )
    df_long = df_long.dropna(subset=['Value'])
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    return df_long

def prepare_master_dataset_corrected(dfs, indicator_names):
    master_list = []
    for df, name in zip(dfs, indicator_names):
        if df.empty: continue
        # Use the corrected function
        df_long = long_format_corrected(df)
        df_long['Indicator Short'] = name
        master_list.append(df_long)
    
    if not master_list:
        return pd.DataFrame()

    full_df = pd.concat(master_list, ignore_index=True)
    max_year = full_df['Year'].max()
    full_df = full_df[full_df['Year'] >= (max_year - 9)]
    return full_df

# --- 3. DATABASE FUNCTIONS ---

def create_database(db_name='world_bank.db'):
    db_path = os.path.join(script_dir, db_name)
    print(f"\n=== DATABASE CREATION ({db_name}) ===")
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS countries 
                 (country_id TEXT PRIMARY KEY, country_name TEXT, region TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS indicators 
                 (indicator_id TEXT PRIMARY KEY, indicator_name TEXT, unit TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS "values" 
                 (country_id TEXT, indicator_id TEXT, year INTEGER, value REAL,
                 FOREIGN KEY(country_id) REFERENCES countries(country_id),
                 FOREIGN KEY(indicator_id) REFERENCES indicators(indicator_id))''')
                 
    conn.commit()
    print("Tables created successfully.")
    return conn

def populate_database(conn, df):
    print("\n=== POPULATING DATABASE ===")
    
    # A. Countries
    countries = df[['Country Code', 'Country Name Standardized', 'Region']].drop_duplicates()
    countries.columns = ['country_id', 'country_name', 'region']
    countries.to_sql('countries', conn, if_exists='replace', index=False)
    
    # B. Indicators
    indicators_data = {
        'indicator_id': ['Nuclear', 'Renew_Prod', 'Renew_Cons', 'GDP', 'Inflation', 'Exports'],
        'indicator_name': ['Nuclear Production', 'Renewable Production', 'Renewable Consumption', 
                           'GDP', 'Inflation', 'Exports'],
        'unit': ['% Total', '% Total', '% Total', 'USD', '% Annual', '% GDP']
    }
    pd.DataFrame(indicators_data).to_sql('indicators', conn, if_exists='replace', index=False)
    
    # C. Values
    values = df[['Country Code', 'Indicator Short', 'Year', 'Value']].copy()
    values.columns = ['country_id', 'indicator_id', 'year', 'value']
    values.to_sql('values', conn, if_exists='replace', index=False)
    print("Data inserted successfully.")

def run_queries(conn):
    print("\n=== RUNNING REQUIRED SQL QUERIES ===")
    
    # Query i
    print("\n[Query i] Yearly Averages per Region (First 5):")
    q1 = '''
    SELECT c.region, v.indicator_id, v.year, AVG(v.value) as avg_val
    FROM "values" v JOIN countries c ON v.country_id = c.country_id
    GROUP BY c.region, v.indicator_id, v.year
    '''
    print(pd.read_sql_query(q1, conn).head())

    # Query ii
    print("\n[Query ii] 10-Year Rolling Avg (Example: USA, GDP):")
    usa = pd.read_sql("SELECT * FROM 'values' WHERE country_id='USA' AND indicator_id='GDP'", conn)
    usa = usa.sort_values('year')
    usa['rolling_10y'] = usa['value'].rolling(10).mean()
    print(usa[['year', 'indicator_id', 'value', 'rolling_10y']].tail())

    # Query iii
    print("\n[Query iii] Top 5 Countries in Europe for Renewable Production:")
    q3 = '''
    SELECT c.country_name, AVG(v.value) as score
    FROM "values" v JOIN countries c ON v.country_id = c.country_id
    WHERE v.year >= (SELECT MAX(year)-9 FROM "values") 
      AND c.region = 'Europe' 
      AND v.indicator_id = 'Renew_Prod'
    GROUP BY c.country_name
    ORDER BY score DESC
    LIMIT 5
    '''
    print(pd.read_sql_query(q3, conn))

# --- EXECUTION ---
if __name__ == "__main__":
    print("Loading Data...")
    try:
        df1 = data_harvesting(get_file_path('nuclear_production.csv'))
        df2 = data_harvesting(get_file_path('renewable_production.csv'))
        df3 = data_harvesting(get_file_path('renewable_consumption.csv'))
        df4 = data_harvesting(get_file_path('gdp.csv'))
        df5 = data_harvesting(get_file_path('inflation.csv'))
        df6 = data_harvesting(get_file_path('exports.csv'))
        
        dfs = [df1, df2, df3, df4, df5, df6]
        names = ["Nuclear", "Renew_Prod", "Renew_Cons", "GDP", "Inflation", "Exports"]
        
        master_df = prepare_master_dataset_corrected(dfs, names)
        
        if not master_df.empty:
            conn = create_database()
            populate_database(conn, master_df)
            run_queries(conn)
            conn.close()
            print("\nPart 2 Complete.")
            
    except Exception as e:
        print(f"Error: {e}")
