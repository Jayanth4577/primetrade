import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(fear_greed_path, historical_path):
    """Loads datasets."""
    logging.info(f"Loading sentiment data from {fear_greed_path}")
    fg_df = pd.read_csv(fear_greed_path)
    logging.info(f"Loading historical trader data from {historical_path}")
    # Using low_memory=False to avoid DtypeWarning on large files
    hist_df = pd.read_csv(historical_path, low_memory=False)
    return fg_df, hist_df

def clean_fear_greed(fg_df):
    """Cleans Fear and Greed Index data."""
    logging.info("Cleaning sentiment data")
    fg_df['date'] = pd.to_datetime(fg_df['date'])
    # Handle duplicates and sort
    fg_df = fg_df.sort_values('date').drop_duplicates(subset=['date'])
    return fg_df

def clean_historical(hist_df):
    """Cleans historical trading data."""
    logging.info("Cleaning historical data")
    
    # Clean column names (strip whitespace)
    hist_df.columns = hist_df.columns.str.strip()
    
    # Parse dates
    # Assuming 'Timestamp IST' format is like '02-12-2024 22:50'
    hist_df['date_time'] = pd.to_datetime(hist_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
    hist_df['date'] = hist_df['date_time'].dt.normalize()
    
    # Handle numeric columns (remove commas if any and convert)
    cols_to_numeric = ['Closed PnL', 'Fee', 'Execution Price', 'Size Tokens', 'Size USD']
    for col in cols_to_numeric:
        if col in hist_df.columns:
            hist_df[col] = pd.to_numeric(hist_df[col].astype(str).str.replace(',', ''), errors='coerce')
            hist_df[col] = hist_df[col].fillna(0)
    
    # Handle missing values
    hist_df = hist_df.dropna(subset=['date_time'])
    
    return hist_df

def merge_datasets(fg_df, hist_df):
    """Merges sentiment data with historical trading data."""
    logging.info("Merging datasets")
    
    merged_df = pd.merge(hist_df, fg_df[['date', 'value', 'classification']], on='date', how='left')
    
    # Forward fill sentiment for any missing dates in between
    merged_df = merged_df.sort_values('date_time')
    merged_df['value'] = merged_df['value'].ffill()
    merged_df['classification'] = merged_df['classification'].ffill()
    
    # Rename columns for clarity
    merged_df = merged_df.rename(columns={'value': 'Sentiment_Value', 'classification': 'Sentiment_Class'})
    
    # Drop rows where sentiment is still NaN (before the first sentiment data point)
    merged_df = merged_df.dropna(subset=['Sentiment_Value'])
    
    return merged_df

def get_profiling_summary(df, name="Dataset"):
    """Returns basic profiling statistics."""
    summary = {
        'Dataset': name,
        'Shape': df.shape,
        'Missing Values': df.isnull().sum().sum(),
        'Duplicates': df.duplicated().sum()
    }
    return summary
