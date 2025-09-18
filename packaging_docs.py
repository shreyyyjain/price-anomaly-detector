import os
import subprocess
import pandas as pd

# Project paths
PROJECT_ROOT = r'C:\Users\shrey\Desktop\Projects\Explainable Price Anomaly Detector for Indian Second-hand Marketplace'
REPORTS_PATH = os.path.join(PROJECT_ROOT, 'reports')
DOCS_PATH = os.path.join(PROJECT_ROOT, 'docs')
os.makedirs(DOCS_PATH, exist_ok=True)

# Generate requirements.txt
def generate_requirements():
    packages = [
        'pandas',
        'numpy',
        'lightgbm',
        'scikit-learn==1.7.2',  # Match your current version
        'matplotlib',
        'shap',
        'streamlit',
        'reportlab'
    ]
    with open(os.path.join(PROJECT_ROOT, 'requirements.txt'), 'w', encoding='utf-8') as f:
        for pkg in packages:
            f.write(f"{pkg}\n")
    print("requirements.txt generated.")

# Create module structure
def create_module_structure():
    module_path = os.path.join(PROJECT_ROOT, 'anomaly_detector')
    os.makedirs(module_path, exist_ok=True)
    
    # Create __init__.py
    with open(os.path.join(module_path, '__init__.py'), 'w', encoding='utf-8') as f:
        f.write("# Anomaly Detector Module\n")
    
    # Copy .py files to module
    scripts = ['automated_report_generation.py', 'evaluation_holdout.py', 'streamlit_demo_ui.py']
    for script in scripts:
        src = os.path.join(PROJECT_ROOT, script)
        dst = os.path.join(module_path, script)
        if os.path.exists(src):
            with open(src, 'r', encoding='utf-8') as f_src, open(dst, 'w', encoding='utf-8') as f_dst:
                f_dst.write(f_src.read())
    
    # Create main.py for module entry point
    main_content = """
from .streamlit_demo_ui import main as run_ui
from .automated_report_generation import generate_report
from .evaluation_holdout import evaluate_models

def main():
    print("Running Price Anomaly Detector...")
    run_ui()  # Launch Streamlit UI

if __name__ == "__main__":
    main()
"""
    with open(os.path.join(module_path, 'main.py'), 'w', encoding='utf-8') as f:
        f.write(main_content)
    
    print("Module structure created at anomaly_detector/")

# Update streamlit_demo_ui.py to work as a module
def update_streamlit_ui():
    ui_path = os.path.join(PROJECT_ROOT, 'anomaly_detector', 'streamlit_demo_ui.py')
    with open(ui_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add main() function for module compatibility
    if "def main():" not in content:
        content = content.replace(
            "from automated_report_generation import generate_report",
            "from .automated_report_generation import generate_report"
        )
        content += """
def main():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle
    import shap
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    import base64
    from io import BytesIO
    from .automated_report_generation import generate_report

    # Project paths
    PROJECT_ROOT = r'C:\\Users\\shrey\\Desktop\\Projects\\Explainable Price Anomaly Detector for Indian Second-hand Marketplace'
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned_engineered.csv')
    ANOMALIES_PATH = os.path.join(PROJECT_ROOT, 'reports', 'validated_anomalies.csv')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'baseline_model.pkl')
    SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'models', 'feature_names.pkl')
    REPORTS_PATH = os.path.join(PROJECT_ROOT, 'reports')

    # Copy the rest of streamlit_demo_ui.py content here
    # Streamlit app
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
"""
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("streamlit_demo_ui.py updated for module compatibility.")

# Create documentation
def create_documentation():
    doc_content = """
# Price Anomaly Detector for Indian Second-hand Marketplace

## Overview
This project detects price anomalies in the Indian second-hand car marketplace using a LightGBM model, SHAP explanations, and business rules. It includes a Streamlit UI for interactive exploration and automated report generation.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/anomaly-detector
   cd Explainable-Price-Anomaly-Detector
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the following directory structure:
   ```
   ├── anomaly_detector/
   │   ├── __init__.py
   │   ├── automated_report_generation.py
   │   ├── evaluation_holdout.py
   │   ├── streamlit_demo_ui.py
   │   ├── main.py
   ├── data/
   │   ├── cleaned_engineered.csv
   ├── models/
   │   ├── baseline_model.pkl
   │   ├── scaler.pkl
   │   ├── feature_names.pkl
   ├── reports/
   │   ├── validated_anomalies.csv
   │   ├── anomaly_report.csv
   │   ├── anomaly_report_summary.csv
   │   ├── scatter_plot.png
   │   ├── shap_summary.png
   │   ├── evaluation_results.csv
   │   ├── precision_recall_comparison.png
   ├── notebooks/
   │   ├── eda.ipynb
   │   ├── feature_engineering_experiments.ipynb
   │   ├── anomaly_scoring.ipynb
   │   ├── shap_explainability.ipynb
   │   ├── business_rules.ipynb
   │   ├── streamlit_demo_skeleton.ipynb
   │   ├── streamlit_demo_ui.ipynb
   ├── requirements.txt
   ├── README.md
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run anomaly_detector/main.py
   ```

## Usage
- **Streamlit UI**: Explore anomalies, filter by rule violations (e.g., high_price_high_km, high_price_old_car), view SHAP plots, and download PDF/CSV reports.
- **Reports**: Generate detailed PDF reports (table, scatter plot, SHAP summary) via the UI or by running `anomaly_detector/automated_report_generation.py`.
- **Evaluation**: Run `anomaly_detector/evaluation_holdout.py` to evaluate model performance on a holdout set with synthetic anomalies.

## Key Results
- Detected 104 validated anomalies (97 high_price_high_km, 7 high_price_old_car).
- LightGBM metrics: Precision=0.6723, Recall=0.3148, F1=0.4288.
- RandomForest metrics: Precision=0.6321, Recall=0.3228, F1=0.4273.
"""
    with open(os.path.join(DOCS_PATH, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(doc_content)
    
    # Update main README
    main_readme_content = f"""
# Project Summary
- Completed packaging and documentation on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}.
- Created anomaly_detector module with main.py entry point.
- Generated requirements.txt and detailed documentation in docs/README.md.
- Next steps: Deploy Streamlit app to a cloud platform.
"""
    with open(os.path.join(PROJECT_ROOT, 'README.md'), 'a', encoding='utf-8') as f:
        f.write(main_readme_content)
    
    print("Documentation created in docs/README.md and main README updated.")

# Execute tasks
generate_requirements()
create_module_structure()
update_streamlit_ui()
create_documentation()