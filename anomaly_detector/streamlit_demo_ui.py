import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import base64
from io import BytesIO
from automated_report_generation import generate_report  # Correct import

# Project paths
PROJECT_ROOT = r'C:\Users\shrey\Desktop\Projects\Explainable Price Anomaly Detector for Indian Second-hand Marketplace'
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned_engineered.csv')
ANOMALIES_PATH = os.path.join(PROJECT_ROOT, 'reports', 'validated_anomalies.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'baseline_model.pkl')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
FEATURE_PATH = os.path.join(PROJECT_ROOT, 'models', 'feature_names.pkl')
REPORTS_PATH = os.path.join(PROJECT_ROOT, 'reports')

def main():
    st.set_page_config(page_title="Price Anomaly Detector", layout="wide")
    st.title("Price Anomaly Detector for Indian Second-hand Marketplace")
    st.markdown("Interactive dashboard to explore price anomalies with SHAP explanations.")

    # Load validated anomalies
    try:
        anomaly_df = pd.read_csv(ANOMALIES_PATH, low_memory=False)
        if anomaly_df.index.name is not None or 'index' in anomaly_df.columns:
            anomaly_df = anomaly_df.reset_index(drop=True)
        anomaly_df['original_index'] = anomaly_df.index if 'original_index' not in anomaly_df.columns else anomaly_df['original_index']
        
        def clean_rule_violations(x):
            if isinstance(x, str) and x.startswith('['):
                try:
                    return ', '.join(eval(x))
                except:
                    return str(x)
            return str(x)
        
        anomaly_df['rule_violations'] = anomaly_df['rule_violations'].apply(clean_rule_violations)
        
        numeric_cols = ['listed_price', 'predicted_price', 'residual', 'km', 'car_age']
        for col in numeric_cols:
            if col in anomaly_df.columns:
                anomaly_df[col] = pd.to_numeric(anomaly_df[col], errors='coerce').astype('float64').fillna(0)
        for col in anomaly_df.columns:
            if col not in numeric_cols + ['original_index']:
                anomaly_df[col] = anomaly_df[col].astype(str).fillna('Unknown')
        
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
    valid_indices = [idx for idx in anomaly_indices if idx in X.index]
    if len(valid_indices) < len(anomaly_indices):
        st.warning(f"{len(anomaly_indices) - len(valid_indices)} invalid indices found. Using {len(valid_indices)} valid anomalies.")
    X_anomalies = X.loc[valid_indices]
    if X_anomalies.empty:
        st.error("No valid anomalies found for SHAP analysis.")
        st.stop()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_anomalies)

    # Layout with columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Anomaly Explorer")
        
        # Extract all unique rules from rule_violations
        all_rules = []
        for violations in anomaly_df['rule_violations']:
            if isinstance(violations, str):
                rules = violations.split(', ')
                all_rules.extend([rule.strip() for rule in rules if rule.strip()])
        
        rule_options = ['All'] + sorted(list(set(all_rules)))
        selected_rule = st.selectbox("Filter by Rule Violation", rule_options)
        sort_by = st.selectbox("Sort By", ['residual', 'listed_price', 'km', 'car_age'])
        ascending = st.checkbox("Sort Ascending", value=False)

        if selected_rule == 'All':
            filtered_anomalies = anomaly_df
        else:
            filtered_anomalies = anomaly_df[anomaly_df['rule_violations'].str.contains(selected_rule, case=False)]
        
        filtered_anomalies = filtered_anomalies.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
        for col in numeric_cols:
            if col in filtered_anomalies.columns:
                filtered_anomalies[col] = pd.to_numeric(filtered_anomalies[col], errors='coerce').astype('float64').fillna(0)
        for col in filtered_anomalies.columns:
            if col not in numeric_cols + ['original_index']:
                filtered_anomalies[col] = filtered_anomalies[col].astype(str).fillna('Unknown')

        st.write(f"**Filtered Anomalies**: {len(filtered_anomalies)}")
        display_cols = ['model', 'oem', 'listed_price', 'predicted_price', 'residual', 'km', 'car_age', 'rule_violations']
        display_df = filtered_anomalies[display_cols].copy()
        for col in ['listed_price', 'predicted_price', 'residual']:
            display_df[col] = display_df[col].apply(lambda x: f'₹{x:,.0f}' if pd.notnull(x) else '₹0')
        for col in ['km', 'car_age']:
            display_df[col] = display_df[col].apply(lambda x: f'{x:,.0f}' if pd.notnull(x) else '0')
        st.dataframe(display_df, height=400)

        if st.button("Generate Report"):
            pdf_path, csv_path = generate_report(filtered_anomalies, X_anomalies, shap_values, explainer)
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            b64 = base64.b64encode(pdf_data).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="anomaly_report.pdf">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.write(f"CSV Report saved at: {csv_path}")

    with col2:
        st.write("**Predicted vs. Actual Prices**")
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.scatter(anomaly_df['listed_price'], anomaly_df['predicted_price'], c='red', alpha=0.5, label='Anomaly')
        plt.plot([anomaly_df['listed_price'].min(), anomaly_df['listed_price'].max()], 
                 [anomaly_df['listed_price'].min(), anomaly_df['listed_price'].max()], 'k--')
        plt.xlabel('Actual Price (₹)')
        plt.ylabel('Predicted Price (₹)')
        plt.title('Predicted vs. Actual Prices (Anomalies)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        st.pyplot(fig)
        plt.close()

        st.write("**SHAP Summary Plot**")
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X_anomalies, show=False)
        plt.title('SHAP Summary for Anomalies')
        st.pyplot(fig)
        plt.close()

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
            
            st.write("**SHAP Force Plot**")
            shap_index = list(X_anomalies.index).index(original_index)
            force_plot = shap.force_plot(
                explainer.expected_value, 
                shap_values[shap_index], 
                X_anomalies.iloc[shap_index], 
                show=False
            )
            st.components.v1.html(shap.getjs() + force_plot.html(), height=400)
            
            st.write("**Feature Values for this Anomaly:**")
            feature_values = X_anomalies.iloc[shap_index]
            important_features = pd.DataFrame({
                'Feature': feature_values.index,
                'Value': feature_values.values
            })
            important_features['Value'] = important_features['Value'].apply(
                lambda x: f'{x:.2f}' if isinstance(x, (int, float)) else str(x)
            )
            st.dataframe(important_features.head(10))
        else:
            st.warning(f"Index {selected_index} not found in dataset.")
    else:
        st.warning("No anomalies available for the selected rule.")

    # Debug information
    with st.expander("Debug Information"):
        st.write(f"SHAP values shape: {np.array(shap_values).shape if hasattr(shap_values, '__len__') else 'N/A'}")
        st.write(f"X_anomalies shape: {X_anomalies.shape}")
        st.write(f"Explainer expected value: {explainer.expected_value}")
        if len(filtered_anomalies) > 0:
            st.write(f"Selected anomaly index in X_anomalies: {original_index in X_anomalies.index}")

if __name__ == "__main__":
    main()
