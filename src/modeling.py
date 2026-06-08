import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def prepare_modeling_data(df):
    """Prepares features and target for predictive modeling."""
    # Target
    y = df['Is_Profitable']
    
    # Features
    features = [
        'Size USD', 'Fee', 'Sentiment_Value', 'Sentiment_Rolling_7d', 
        'Sentiment_Momentum_7d', 'Win_Rate_Trader', 'Average_PnL_Trader'
    ]
    
    # Keep only available columns
    features = [f for f in features if f in df.columns]
    
    X = df[features].copy()
    
    # Fill any remaining NaNs in features
    X = X.fillna(X.median())
    
    # Encoding Categorical if we add side or direction
    if 'Direction' in df.columns:
        le = LabelEncoder()
        X['Direction_Enc'] = le.fit_transform(df['Direction'].astype(str))
    
    return X, y

def train_and_evaluate_models(X, y):
    """Trains Logistic Regression, Random Forest, and XGBoost models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        fitted_models[name] = model
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        
    return results, fitted_models, X_train.columns

def plot_feature_importance(fitted_models, feature_names, output_dir='reports/figures'):
    """Plots feature importance for Random Forest and XGBoost."""
    os.makedirs(output_dir, exist_ok=True)
    
    for name in ['Random Forest', 'XGBoost']:
        if name in fitted_models:
            model = fitted_models[name]
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
            plt.title(f'Feature Importance ({name})')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/feature_importance_{name.replace(" ", "_")}.png', dpi=300)
            plt.close()
