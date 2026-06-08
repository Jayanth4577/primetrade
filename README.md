# 📈 Crypto Trader Performance vs. Market Sentiment Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%23150458.svg?style=flat)](https://xgboost.readthedocs.io/)

> A consulting-grade Data Science assignment exploring how extreme market emotions (Fear & Greed) influence individual trader behavior, risk management, and profitability. 

---

## 📑 Table of Contents
- [Business Overview](#-business-overview)
- [Methodology](#-methodology)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Results & Modeling](#-results--modeling)
- [Visualizations](#-visualizations)

---

## 💼 Business Overview
Cryptocurrency markets are highly volatile and heavily influenced by retail sentiment. This project investigates whether extreme market emotions—quantified by the **Bitcoin Fear & Greed Index**—adversely affect individual trader profitability and sizing behavior. 

By identifying these behavioral traps, trading platforms can implement dynamic margin limits, issue targeted risk warnings, and construct contrarian automated market-making algorithms.

---

## 🔬 Methodology

1. **Data Processing**: Parsed disorganized trade timestamps, cleaned monetary strings, and merged high-frequency historical trading data with daily sentiment scores using forward-filling.
2. **Feature Engineering**: Engineered market context features (7-day sentiment momentum, rolling averages) and synthesized historical "Trader Skill Profiles" (win rates, median PnL).
3. **Statistical Testing**: Proved the statistical significance between market sentiment and trade profitability using **ANOVA** and **Chi-Square** contingency tests.
4. **Segmentation**: Used **K-Means Clustering & PCA** to group traders into actionable segments (e.g., *High-Risk Momentum Traders*, *Consistent Scalpers*).
5. **Predictive Modeling**: Framed profitability as a binary classification target and trained **Random Forest** and **XGBoost** models to predict successful trades, generating clear feature importance outputs.

---

## ⚡ Key Features

- **Automated End-to-End Pipeline**: A singular Jupyter Notebook (`Main_Analysis.ipynb`) orchestrates data cleaning, visualization, statistical proofs, and machine learning.
- **Consulting-Grade Reporting**: Includes an Executive Summary (`reports/Executive_Report.md`) and Top 10 Actionable Insights (`reports/insights.txt`).
- **Modular Codebase**: Object-oriented, segmented Python scripts (`src/`) that keep the notebook clean and readable.

---

## 📁 Project Structure

```text
├── data/                             # Ignored CSV datasets (Place fear_greed_index.csv & historical_data.csv here)
├── notebooks/
│   └── Main_Analysis.ipynb           # Main execution notebook
├── reports/
│   ├── figures/                      # Auto-generated PNG visualizations
│   ├── Executive_Report.md           # Professional summary
│   └── insights.txt                  # Top 10 actionable business insights
├── src/
│   ├── analysis.py                   # Statistical testing and KMeans clustering
│   ├── data_processing.py            # Data loading, cleaning, and merging logic
│   ├── eda.py                        # Automated seaborn/matplotlib pipelines
│   ├── features.py                   # Feature engineering logic
│   └── modeling.py                   # RandomForest and XGBoost modeling pipelines
├── .gitignore
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/primetrade.git
   cd primetrade
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Datasets:**
   Ensure `fear_greed_index.csv` and `historical_data.csv` are placed in the root directory.

5. **Run the Analysis:**
   Launch Jupyter Notebook and open `notebooks/Main_Analysis.ipynb`.
   ```bash
   jupyter notebook
   ```

---

## 🤖 Results & Modeling
Our tree-based machine learning models proved highly capable of predicting a trade's success by combining macro-market sentiment with micro-trader history:

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| **Logistic Regression** | 76.7% | 0.73 | 0.684 |
| **Random Forest** | 94.8% | 0.93 | **0.986** |
| **XGBoost** | 94.2% | 0.93 | 0.985 |

*Key finding: The interaction between a trader's historical win rate and the 7-day sentiment momentum was the strongest indicator of success.*

---

## 📊 Visualizations
*Run the Jupyter Notebook to automatically generate and populate your local `reports/figures/` folder with deep-dive visual insights, including correlation heatmaps, PnL boxplots, and trader segmentation clusters.*
