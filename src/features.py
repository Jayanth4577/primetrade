import pandas as pd
import numpy as np

def create_sentiment_features(df):
    """Engineers features related to market sentiment."""
    # Ordinal encoding of sentiment
    sentiment_map = {
        'Extreme Fear': 0,
        'Fear': 1,
        'Neutral': 2,
        'Greed': 3,
        'Extreme Greed': 4
    }
    df['Sentiment_Score'] = df['Sentiment_Class'].map(sentiment_map)
    
    # We need a daily dataframe to calculate rolling metrics properly
    daily_sentiment = df[['date', 'Sentiment_Value', 'Sentiment_Score']].drop_duplicates('date').sort_values('date')
    
    # Rolling averages
    daily_sentiment['Sentiment_Rolling_7d'] = daily_sentiment['Sentiment_Value'].rolling(window=7, min_periods=1).mean()
    daily_sentiment['Sentiment_Rolling_30d'] = daily_sentiment['Sentiment_Value'].rolling(window=30, min_periods=1).mean()
    
    # Momentum (difference between today and 7 days ago)
    daily_sentiment['Sentiment_Momentum_7d'] = daily_sentiment['Sentiment_Value'].diff(periods=7).fillna(0)
    
    # Merge back to main dataframe
    df = pd.merge(df, daily_sentiment.drop(columns=['Sentiment_Value', 'Sentiment_Score']), on='date', how='left')
    
    return df

def create_trader_features(df):
    """Engineers features related to individual trader performance."""
    # Target variable: Is trade profitable?
    df['Is_Profitable'] = (df['Closed PnL'] > 0).astype(int)
    
    # Calculate Leverage if Size USD and Margin are available (Approximate logic if margin is missing)
    # Assuming standard if we don't have explicit margin, we might skip or approximate.
    # We will create basic trade metrics.
    df['Trade_Value_Category'] = pd.qcut(df['Size USD'].rank(method='first'), q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Aggregate features per account (Trader Profiling)
    trader_profile = df.groupby('Account').agg(
        Total_Trades=('Order ID', 'count'),
        Win_Rate=('Is_Profitable', 'mean'),
        Average_PnL=('Closed PnL', 'mean'),
        Median_PnL=('Closed PnL', 'median'),
        Total_Volume_USD=('Size USD', 'sum'),
        Average_Trade_Size=('Size USD', 'mean')
    ).reset_index()
    
    # Merge trader profile back to evaluate trade in context of trader's overall skill
    df = pd.merge(df, trader_profile, on='Account', how='left', suffixes=('', '_Trader'))
    
    return df

def run_feature_engineering(df):
    """Runs all feature engineering pipelines."""
    df = create_sentiment_features(df)
    df = create_trader_features(df)
    return df
