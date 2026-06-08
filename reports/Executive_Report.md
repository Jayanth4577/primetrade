# Executive Report: Trader Performance vs. Market Sentiment

## 1. Executive Summary
This report investigates how Bitcoin market sentiment (quantified by the Fear & Greed Index) influences trader behavior and profitability. By merging historical trading data with daily sentiment indices, we identified distinct patterns indicating that extreme market sentiments significantly alter risk-taking and win rates. We built predictive models (XGBoost, Random Forest) that successfully identify whether a trade will be profitable based on market context and trader historical profiles.

## 2. Business Understanding
Trading platforms and funds can leverage sentiment analysis to adjust dynamic margins, suggest risk limits to users, or build automated market-making strategies that counteract retail behavioral biases.

## 3. Data Overview & Methodology
- **Datasets**: `historical_data.csv` (Trade executions) and `fear_greed_index.csv` (Daily sentiment scores 0-100).
- **Processing**: We aligned the disparate timestamps by down-sampling sentiment to daily and merging it with trade timestamps. Forward filling was used to ensure gapless sentiment tracking.
- **Engineering**: We created features like 7-day Sentiment Momentum, Rolling Averages, and Trader-specific metrics (Historical Win Rate, Average PnL).

## 4. Key Findings (EDA & Stats)
1. **Profitability Under Extremes**: Win rates tend to dip during 'Extreme Greed' as retail traders often buy the top, leading to negative Closed PnLs upon mean reversion.
2. **Size and Sentiment**: Median position sizes are significantly larger during 'Greed' phases compared to 'Extreme Fear'.
3. **Statistical Significance**: ANOVA and Kruskal-Wallis tests confirm that the distribution of Closed PnL is statistically different across the 5 sentiment classifications (p < 0.05).

## 5. Trader Segmentation
Using K-Means clustering on aggregated trader profiles, we identified 4 segments:
- **High-Risk Momentum Traders**: High average trade size, low win rate.
- **Consistent Scalpers**: High frequency, low median PnL, high win rate.
- **Sentiment-Sensitive Traders**: Highly active only during Greed phases.
- **Contrarians**: Profitable primarily during Fear phases.

## 6. Predictive Modeling
We framed profitability prediction as a binary classification problem (`Closed PnL > 0`).
- **Models Used**: Logistic Regression, Random Forest, XGBoost.
- **Performance**: Tree-based models out-performed Logistic Regression. XGBoost achieved strong ROC-AUC scores by capturing non-linear relationships between Trader historical skill (`Win_Rate_Trader`) and `Sentiment_Rolling_7d`.
- **Feature Importance**: Trader's historical win rate and average trade size were the strongest predictors, followed closely by the 7-day sentiment momentum.

## 7. Recommendations
1. **Risk Warnings**: Trigger UI risk warnings for users who dramatically increase leverage during 'Extreme Greed'.
2. **Automated Trading**: Develop a contrarian systematic strategy that fades aggressive retail long positions during peak 'Extreme Greed'.
3. **User Education**: Implement customized educational nudges for 'High-Risk' clustered users.

## 8. Limitations & Future Work
- **Granularity**: The Fear & Greed index is daily, while crypto trades 24/7. Intraday sentiment metrics (e.g., from Twitter/X API) would yield sharper signals.
- **Slippage & Funding**: Analysis currently assumes gross execution price without modeling granular funding rates for perpetual futures.
