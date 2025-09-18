import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Project paths
PROJECT_ROOT = r'C:\Users\shrey\Desktop\Projects\Explainable Price Anomaly Detector for Indian Second-hand Marketplace'
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned_engineered.csv')
ANOMALIES_PATH = os.path.join(PROJECT_ROOT, 'reports', 'validated_anomalies.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'baseline_model.pkl')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
FEATURE_PATH = os.path.join(PROJECT_ROOT, 'models', 'feature_names.pkl')
REPORTS_PATH = os.path.join(PROJECT_ROOT, 'reports')

# Streamlit app
st.title("Price Anomaly Detector for Indian Second-hand Marketplace")
st.markdown("Explore validated price anomalies with SHAP explanations.")

# Load validated anomalies
try:
    anomaly_df = pd.read_csv(ANOMALIES_PATH, low_memory=False)
    # Reset index to simple integers and drop any existing index column
    if anomaly_df.index.name is not None or 'index' in anomaly_df.columns:
        anomaly_df = anomaly_df.reset_index(drop=True)
    # Store original indices for SHAP
    anomaly_df['original_index'] = anomaly_df.index
    
    # Convert rule_violations to clean string
    def clean_rule_violations(x):
        if isinstance(x, str) and x.startswith('['):
            try:
                return ', '.join(eval(x))
            except:
                return str(x)
        return str(x)
    
    anomaly_df['rule_violations'] = anomaly_df['rule_violations'].apply(clean_rule_violations)
    
    # Convert columns to Arrow-compatible types
    numeric_cols = ['listed_price', 'predicted_price', 'residual', 'km', 'car_age']
    for col in numeric_cols:
        if col in anomaly_df.columns:
            anomaly_df[col] = pd.to_numeric(anomaly_df[col], errors='coerce').astype('float64').fillna(0)
    
    # Convert all other columns to string
    for col in anomaly_df.columns:
        if col not in numeric_cols + ['original_index']:
            anomaly_df[col] = anomaly_df[col].astype(str).fillna('Unknown')
    
    # Debug types (uncomment if needed)
    # st.write("Anomaly DataFrame types:", anomaly_df.dtypes)
    # st.write("Sample anomaly data:", anomaly_df.head())
    st.write(f"**Total Validated Anomalies**: {len(anomaly_df)}")
except FileNotFoundError:
    st.error(f"File not found: {ANOMALIES_PATH}")
    st.stop()

# Load full dataset
try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
except FileNotFoundError:
    st.error(f"File not found: {DATA_PATH}")
    st.stop()

# Load model, scaler, feature names
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(FEATURE_PATH, 'rb') as f:
        feature_names = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Model or scaler file not found: {e}")
    st.stop()

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

# Prepare features
X = df[num_cols + cat_cols].copy()
for col in cat_cols:
    X[col] = X[col].astype('category')
X[num_cols] = scaler.transform(X[num_cols])

# SHAP for anomalies only
anomaly_indices = anomaly_df['original_index'].tolist()
X_anomalies = X.loc[anomaly_indices]
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_anomalies)
except Exception as e:
    st.error(f"Error initializing SHAP explainer: {e}")
    st.stop()

# Filter by rule violations
st.subheader("Filter Anomalies by Rule Violations")
rule_options = ['All', 'high_price_high_km', 'high_price_old_car']
selected_rule = st.selectbox("Select Rule Violation", rule_options)

if selected_rule == 'All':
    filtered_anomalies = anomaly_df
else:
    filtered_anomalies = anomaly_df[anomaly_df['rule_violations'].str.contains(selected_rule, case=False)]

# Ensure filtered_anomalies is Arrow-compatible
filtered_anomalies = filtered_anomalies.copy().reset_index(drop=True)
for col in numeric_cols:
    if col in filtered_anomalies.columns:
        filtered_anomalies[col] = pd.to_numeric(filtered_anomalies[col], errors='coerce').astype('float64').fillna(0)
for col in filtered_anomalies.columns:
    if col not in numeric_cols + ['original_index']:
        filtered_anomalies[col] = filtered_anomalies[col].astype(str).fillna('Unknown')

# Debug types (uncomment if needed)
# st.write("Filtered DataFrame types:", filtered_anomalies.dtypes)
st.write(f"**Filtered Anomalies**: {len(filtered_anomalies)}")

# Display anomaly table
st.subheader("Anomaly Details")
display_cols = ['model', 'oem', 'listed_price', 'predicted_price', 'residual', 'km', 'car_age', 'rule_violations']
display_df = filtered_anomalies[display_cols].copy()
for col in ['listed_price', 'predicted_price', 'residual']:
    display_df[col] = display_df[col].apply(lambda x: f'₹{x:,.0f}' if pd.notnull(x) else '₹0')
for col in ['km', 'car_age']:
    display_df[col] = display_df[col].apply(lambda x: f'{x:,.0f}' if pd.notnull(x) else '0')
st.dataframe(display_df)

# SHAP explanation
st.subheader("SHAP Explanation for Selected Anomaly")
if len(filtered_anomalies) > 0:
    selected_index = st.selectbox("Select Anomaly Index", filtered_anomalies.index)
    original_index = filtered_anomalies.loc[selected_index, 'original_index']
    if original_index in X_anomalies.index:
        st.write(f"**Details for Anomaly (Index {selected_index})**")
        anomaly_details = filtered_anomalies.loc[selected_index, display_cols].copy()
        anomaly_details_display = pd.DataFrame([anomaly_details])
        for col in ['listed_price', 'predicted_price', 'residual']:
            anomaly_details_display[col] = anomaly_details_display[col].apply(lambda x: f'₹{x:,.0f}' if pd.notnull(x) else '₹0')
        for col in ['km', 'car_age']:
            anomaly_details_display[col] = anomaly_details_display[col].apply(lambda x: f'{x:,.0f}' if pd.notnull(x) else '0')
        st.dataframe(anomaly_details_display)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        shap_index = anomaly_indices.index(original_index)
        shap.force_plot(explainer.expected_value, shap_values[shap_index], X_anomalies.loc[original_index], matplotlib=True, show=False)
        plt.title(f'SHAP Force Plot for Anomaly (Index {selected_index})')
        st.pyplot(fig)
        plt.close()
    else:
        st.warning(f"Index {selected_index} not found in dataset.")
else:
    st.warning("No anomalies available for the selected rule.")

# Update README
readme_content = f"""
# Streamlit Demo Skeleton Summary
- Built a Streamlit app to display {len(anomaly_df)} validated anomalies from business rules.
- Features: Filter by rule violations (high_price_high_km, high_price_old_car), view anomaly details, and display SHAP force plots.
- Resolved Arrow serialization issues for DataFrame display.
- Next steps: Polish UI with additional visuals and interactive features.
"""
with open(os.path.join(PROJECT_ROOT, 'README.md'), 'a', encoding='utf-8') as f:
    f.write(readme_content)
st.write('README.md updated with Streamlit demo skeleton summary.')