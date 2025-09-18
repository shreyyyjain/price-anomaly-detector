
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
