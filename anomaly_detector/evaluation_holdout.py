import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Project paths
DATA_PATH = 'data/cleaned_engineered.csv'
ANOMALIES_PATH = 'reports/validated_anomalies.csv'
MODEL_PATH = 'models/baseline_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURE_PATH = 'models/feature_names.pkl'
REPORTS_PATH = 'reports'
os.makedirs(REPORTS_PATH, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip().str.lower()

# Load model, scaler, and feature names
with open(MODEL_PATH, 'rb') as f:
    lgb_model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(FEATURE_PATH, 'rb') as f:
    feature_names = pickle.load(f)

# Feature engineering
df['log_price'] = np.log1p(df['listed_price'])
mean_global = df['log_price'].mean()
k = 5
for col in ['oem', 'model', 'city']:
    target_mean = df.groupby(col)['log_price'].mean()
    count = df.groupby(col)['log_price'].count()
    smooth = (target_mean * count + mean_global * k) / (count + k)
    df[f'{col}_target_enc'] = df[col].map(smooth)
for col in ['oem', 'model', 'city']:
    freq = df[col].value_counts()
    df[f'{col}_freq_enc'] = df[col].map(freq)
df['brand_age'] = df['car_age'] * df['oem_target_enc']
df['km_per_year_age'] = df['km_per_year'] * df['car_age']
df['power_weight_ratio'] = df['max power delivered'] / df['kerb weight']

# Define features
num_cols = [
    'km', 'car_age', 'km_per_year', 'max power delivered', 'alloy wheel size',
    'length', 'width', 'height', 'wheel base', 'front tread', 'rear tread',
    'kerb weight', 'gross weight', 'top speed', 'acceleration', 'bore',
    'oem_target_enc', 'model_target_enc', 'city_target_enc',
    'brand_age', 'km_per_year_age', 'power_weight_ratio'
]
cat_cols = [
    'transmission', 'fuel', 'owner_type', 'drive type', 'steering type',
    'front brake type', 'rear brake type', 'tyre type'
]
num_cols = [col for col in num_cols if col in df.columns]
cat_cols = [col for col in cat_cols if col in df.columns]

# Prepare features and target - KEEP ORIGINAL FOR LIGHTGBM
X_original = df[num_cols + cat_cols].copy()
for col in cat_cols:
    X_original[col] = X_original[col].astype('category')
X_original[num_cols] = scaler.transform(X_original[num_cols])
y = df['log_price']

# CREATE ENCODED VERSION FOR RANDOMFOREST
X_encoded = X_original.copy()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_original[col].astype(str))
    label_encoders[col] = le

# Use X_original for LightGBM and X_encoded for RandomForest
X_train_orig, X_holdout_orig, y_train, y_holdout = train_test_split(X_original, y, test_size=0.2, random_state=42)
X_train_enc, X_holdout_enc, _, _ = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
holdout_indices = X_holdout_orig.index
holdout_df = df.loc[holdout_indices].copy()

# Generate synthetic anomalies (e.g., 5% of holdout set)
np.random.seed(42)
n_anomalies = int(0.05 * len(X_holdout_orig))
anomaly_indices = np.random.choice(X_holdout_orig.index, size=n_anomalies, replace=False)
holdout_df.loc[anomaly_indices, 'listed_price'] *= np.random.uniform(0.1, 3.0, size=n_anomalies)  # Random price scaling
holdout_df['is_anomaly'] = 0
holdout_df.loc[anomaly_indices, 'is_anomaly'] = 1
y_holdout_anomalies = np.log1p(holdout_df['listed_price'])

# Predict with LightGBM (using original categorical data)
y_pred_lgb = lgb_model.predict(X_holdout_orig)
y_pred_lgb_actual = np.expm1(y_pred_lgb)
y_actual = np.expm1(y_holdout_anomalies)
residuals_lgb = np.abs(y_actual - y_pred_lgb_actual)

# Flag anomalies (residuals > 2 standard deviations)
residual_mean = residuals_lgb.mean()
residual_std = residuals_lgb.std()
anomaly_threshold = residual_mean + 2 * residual_std
predicted_anomalies_lgb = (residuals_lgb > anomaly_threshold).astype(int)

# Train RandomForest for comparison (using encoded data)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train_enc, y_train)
y_pred_rf = rf_model.predict(X_holdout_enc)
y_pred_rf_actual = np.expm1(y_pred_rf)
residuals_rf = np.abs(y_actual - y_pred_rf_actual)
predicted_anomalies_rf = (residuals_rf > anomaly_threshold).astype(int)

# Evaluate performance
true_anomalies = holdout_df['is_anomaly']
metrics = {
    'LightGBM': {
        'Precision': precision_score(true_anomalies, predicted_anomalies_lgb),
        'Recall': recall_score(true_anomalies, predicted_anomalies_lgb),
        'F1': f1_score(true_anomalies, predicted_anomalies_lgb)
    },
    'RandomForest': {
        'Precision': precision_score(true_anomalies, predicted_anomalies_rf),
        'Recall': recall_score(true_anomalies, predicted_anomalies_rf),
        'F1': f1_score(true_anomalies, predicted_anomalies_rf)
    }
}

# Print metrics
print("Evaluation Metrics:")
for model, scores in metrics.items():
    print(f"{model}:")
    print(f"  Precision: {scores['Precision']:.4f}")
    print(f"  Recall: {scores['Recall']:.4f}")
    print(f"  F1: {scores['F1']:.4f}")

# Save evaluation results
results_df = pd.DataFrame({
    'Model': ['LightGBM', 'RandomForest'],
    'Precision': [metrics['LightGBM']['Precision'], metrics['RandomForest']['Precision']],
    'Recall': [metrics['LightGBM']['Recall'], metrics['RandomForest']['Recall']],
    'F1': [metrics['LightGBM']['F1'], metrics['RandomForest']['F1']]
})
results_df.to_csv(os.path.join(REPORTS_PATH, 'evaluation_results.csv'), index=False)

# Plot precision-recall comparison
plt.figure(figsize=(8, 6))
models = ['LightGBM', 'RandomForest']
precisions = [metrics[model]['Precision'] for model in models]
recalls = [metrics[model]['Recall'] for model in models]
plt.bar(models, precisions, width=0.35, label='Precision', color='blue')
plt.bar([m + 0.35 for m in range(len(models))], recalls, width=0.35, label='Recall', color='orange')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Precision and Recall Comparison')
plt.legend()
plt.savefig(os.path.join(REPORTS_PATH, 'precision_recall_comparison.png'))
plt.close()

# Update README
readme_content = f"""
# Evaluation Summary
- Evaluated anomaly detection on a holdout set with {n_anomalies} synthetic anomalies.
- Metrics:
  - LightGBM: Precision={metrics['LightGBM']['Precision']:.4f}, Recall={metrics['LightGBM']['Recall']:.4f}, F1={metrics['LightGBM']['F1']:.4f}
  - RandomForest: Precision={metrics['RandomForest']['Precision']:.4f}, Recall={metrics['RandomForest']['Recall']:.4f}, F1={metrics['RandomForest']['F1']:.4f}
- Results saved to reports/evaluation_results.csv.
- Precision-recall comparison plot saved to reports/precision_recall_comparison.png.
- Next steps: Package the project and prepare documentation.
"""
with open(('README.md'), 'a', encoding='utf-8') as f:
    f.write(readme_content)
print('README.md updated with evaluation summary.')