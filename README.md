# Trader Performance & Bitcoin Sentiment Analysis

## Overview
This repository contains a complete, consulting-grade data science analysis exploring the relationship between individual trader performance and overall Bitcoin market sentiment (Fear & Greed Index).

## Project Structure
- `data/`: (Not included in version control) Put `fear_greed_index.csv` and `historical_data.csv` here.
- `notebooks/`: Contains `Main_Analysis.ipynb` for the end-to-end execution flow.
- `src/`: Reusable Python modules (`data_processing.py`, `features.py`, `eda.py`, `analysis.py`, `modeling.py`).
- `reports/`: Contains the final `Executive_Report.md`, generated figures, and insights.

## Setup Instructions

1. **Clone the repository.**
2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Analysis**:
   - Place the datasets (`fear_greed_index.csv` and `historical_data.csv`) in the root directory (or update paths in the notebook).
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open `notebooks/Main_Analysis.ipynb` and run all cells.

## Methodology
1. **Data Processing**: Parsed timestamps, aligned dates, and merged daily sentiment to high-frequency trades.
2. **Feature Engineering**: Generated momentum, rolling sentiment, and trader historical skill profiles.
3. **EDA & Stats**: Explored PnL distributions by sentiment classes. Tested significance using ANOVA and Chi-Square.
4. **Modeling**: Built Random Forest and XGBoost classifiers to predict whether a specific trade will end in a positive PnL.
