import os
import glob
import pandas as pd

# Target 10 FTSE 100 tickers: Top 7 Mega-caps + 3 Lower FTSE 100 Constituents
# Updated 10 FTSE 100 Stock Portfolio Subset
target_tickers = [
    # 7 Large Cap (Big)
    'HSBA',  # HSBC Holdings
    'SHEL',  # Shell
    'AZN',   # AstraZeneca
    'RR.',   # Rolls-Royce Holdings
    'ULVR',  # Unilever
    'RIO',   # Rio Tinto
    'BP.',   # BP
    
    # 3 Small Cap (Small)
    'JD.',   # JD Sports Fashion (also 'JD' depending on data feed)
    'AUTO',  # Auto Trader Group
    'BRBY'   # Burberry Group
]

def load_and_combine_target_stocks(archive_dir: str, tickers: list, start_date: str = '2024-01-01') -> pd.DataFrame:
    """
    Loops through all individual stock CSV files in the archive folder,
    filters for target tickers (or fallbacks like JD), cleans N/As and zeroes,
    and combines them into a single dataframe.
    """
    all_csvs = glob.glob(os.path.join(archive_dir, "*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"No CSV files found inside directory: '{archive_dir}'")

    print(f"Scanning {len(all_csvs)} stock files in '{archive_dir}'...")

    dataframes = []
    found_tickers = set()

    for file_path in all_csvs:
        filename = os.path.basename(file_path)
        symbol = os.path.splitext(filename)[0].upper().replace('.L', '')
        
        # Match symbol against target list
        matched_target = None
        for t in tickers:
            clean_t = t.upper().replace('.', '')
            clean_symbol = symbol.replace('.', '')
            if clean_symbol == clean_t:
                matched_target = t
                break

        if matched_target and matched_target not in found_tickers:
            print(f" -> Processing target stock file: {filename} (Mapped to: {matched_target})")
            df = pd.read_csv(file_path)
            
            # Standardise column headers
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Set explicit ticker column
            df['ticker'] = matched_target
            
            # Parse date and filter post-2024
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= start_date].copy()
            
            # Drop NaNs and invalid prices
            df = df.dropna(subset=['date', 'close'])
            df = df[df['close'] > 0]
            
            # Drop duplicate dates
            df = df.drop_duplicates(subset=['date'])
            
            dataframes.append(df)
            found_tickers.add(matched_target)

    if not dataframes:
        raise ValueError("None of the target tickers were found in the archive folder.")

    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Cap at top 10 unique tickers if more were found
    unique_found = combined_df['ticker'].unique()
    if len(unique_found) > 10:
        selected_10 = unique_found[:10]
        combined_df = combined_df[combined_df['ticker'].isin(selected_10)].copy()

    combined_df = combined_df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    
    print(f"\nSuccessfully combined {len(combined_df)} clean rows across {combined_df['ticker'].nunique()} tickers:")
    print(list(combined_df['ticker'].unique()))
    return combined_df

if __name__ == "__main__":
    archive_folder = "archive"
    
    try:
        clean_df = load_and_combine_target_stocks(archive_folder, target_tickers)
        
        output_file = "clean_ftse100_subset.csv"
        clean_df.to_csv(output_file, index=False)
        print(f"\nClean dataset saved as '{output_file}'.")
        
    except Exception as e:
        print(f"Error: {e}")
