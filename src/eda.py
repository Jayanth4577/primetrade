import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_style():
    """Sets a professional consulting-grade visualization style."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'figure.figsize': (10, 6),
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

def plot_pnl_by_sentiment(df, output_dir='reports/figures'):
    """Plots PnL distribution under different sentiment classifications."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    # Define order for sentiment
    order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    
    # Boxplot for PnL
    sns.boxplot(x='Sentiment_Class', y='Closed PnL', data=df, order=order, showfliers=False, palette='coolwarm')
    plt.title('Distribution of Closed PnL by Market Sentiment (Excl. Outliers)')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Closed PnL ($)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pnl_by_sentiment.png', dpi=300)
    plt.close()

def plot_win_rate_by_sentiment(df, output_dir='reports/figures'):
    """Plots Win Rate under different sentiment classifications."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    win_rates = df.groupby('Sentiment_Class')['Is_Profitable'].mean().reindex(order).reset_index()
    
    sns.barplot(x='Sentiment_Class', y='Is_Profitable', data=win_rates, palette='viridis')
    plt.title('Trader Win Rate by Market Sentiment')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Win Rate (%)')
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50% Baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/win_rate_by_sentiment.png', dpi=300)
    plt.close()

def plot_trade_size_by_sentiment(df, output_dir='reports/figures'):
    """Plots average trade size behavior under different sentiments."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    
    sns.barplot(x='Sentiment_Class', y='Size USD', data=df, order=order, estimator=np.median, palette='magma')
    plt.title('Median Position Size (USD) by Market Sentiment')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Median Size (USD)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/trade_size_by_sentiment.png', dpi=300)
    plt.close()

def plot_correlation_matrix(df, output_dir='reports/figures'):
    """Plots a correlation matrix of numerical features."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Select subset of interesting columns to avoid huge matrix
    interest_cols = ['Closed PnL', 'Size USD', 'Fee', 'Sentiment_Value', 
                     'Sentiment_Rolling_7d', 'Sentiment_Momentum_7d', 
                     'Win_Rate', 'Average_PnL', 'Total_Volume_USD']
    
    valid_cols = [c for c in interest_cols if c in num_cols]
    corr = df[valid_cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300)
    plt.close()

def run_all_eda(df, output_dir='reports/figures'):
    """Executes the full EDA pipeline."""
    set_style()
    plot_pnl_by_sentiment(df, output_dir)
    plot_win_rate_by_sentiment(df, output_dir)
    plot_trade_size_by_sentiment(df, output_dir)
    plot_correlation_matrix(df, output_dir)
    print(f"EDA visualizations saved to {output_dir}")
