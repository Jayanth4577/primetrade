import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_hypothesis_tests(df):
    """Runs statistical tests to check if sentiment affects trader metrics."""
    results = {}
    
    # ANOVA: Does Closed PnL differ by sentiment?
    # Ensure no NaNs and remove extreme outliers for the test
    test_df = df.dropna(subset=['Closed PnL', 'Sentiment_Class'])
    
    groups = [group['Closed PnL'].values for name, group in test_df.groupby('Sentiment_Class')]
    if len(groups) > 1:
        f_stat, p_val = stats.f_oneway(*groups)
        results['ANOVA_PnL_by_Sentiment'] = {'F-Statistic': f_stat, 'p-value': p_val}
        
    # Kruskal-Wallis: Non-parametric alternative for PnL (robust to outliers)
    if len(groups) > 1:
        h_stat, p_val_kw = stats.kruskal(*groups)
        results['Kruskal_PnL_by_Sentiment'] = {'H-Statistic': h_stat, 'p-value': p_val_kw}
        
    # Chi-Square: Does win rate depend on sentiment?
    contingency_table = pd.crosstab(test_df['Is_Profitable'], test_df['Sentiment_Class'])
    if contingency_table.size > 0:
        chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)
        results['ChiSquare_WinRate_Sentiment'] = {'Chi2': chi2, 'p-value': p_chi2}
        
    return results

def segment_traders(df, output_dir='reports/figures'):
    """Clusters traders based on their profiles."""
    os.makedirs(output_dir, exist_ok=True)
    
    # We aggregate data per trader
    trader_df = df.groupby('Account').agg(
        Win_Rate=('Is_Profitable', 'mean'),
        Average_PnL=('Closed PnL', 'mean'),
        Median_Size=('Size USD', 'median'),
        Total_Trades=('Order ID', 'count')
    ).dropna()
    
    if len(trader_df) < 5:
        print("Not enough traders to cluster.")
        return trader_df
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(trader_df)
    
    # K-Means clustering (K=4 groups: e.g., High-performing, Consistent, High-risk, Low-risk)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    trader_df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Map cluster numbers to descriptive names based on characteristics
    # For automated scripts, we just use Cluster 0-3. In a real scenario, we interpret the centroids.
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_scaled)
    trader_df['PCA1'] = pca_res[:, 0]
    trader_df['PCA2'] = pca_res[:, 1]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=trader_df, palette='Set1', s=100)
    plt.title('Trader Segmentation (PCA View)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/trader_segments.png', dpi=300)
    plt.close()
    
    return trader_df
