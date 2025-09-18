Explainable Price Anomaly Detector for Indian Second-hand Car Marketplace
=========================================================================

📊 Project Overview
-------------------

This project develops an explainable machine learning system to detect overpriced and underpriced used cars in the Indian second-hand market using the CarDekho dataset (37,813 listings, 67 columns). It identifies **103 validated anomalies** (96 `high_price_high_km`, 6 `high_price_old_car`, 1 combined) through LightGBM predictions, SHAP explanations, and business rules. The system is deployed as an interactive Streamlit app at [price-anomaly-detector.streamlit.app](https://price-anomaly-detector.streamlit.app), offering dynamic filtering, visualizations, and downloadable PDF/CSV reports.

**Business Impact**: Empowers buyers to avoid overpaying (e.g., 96 high-mileage cars priced ₹1M+) and sellers to price competitively, enhancing trust in platforms like CarDekho.

**Tech Stack**: Python, Pandas, LightGBM, SHAP, Streamlit, ReportLab, Git LFS.

📈 Dataset
----------

*   **Source**: CarDekho used cars (India).
*   **Size**: 37,813 rows, 67 columns initially.
*   **Key Features**:
    *   `listed_price`: Mean ₹799,986, clipped to ₹84,696–₹4,872,000 (right-skewed).
    *   `km`: Mean 62,409, clipped to 3,358–191,783.
    *   `myear`: Car age mean 9.43 years (2025 - `myear`).
    *   `fuel`, `transmission`, `oem`, `model`, `city`: High-cardinality categoricals.
*   **Challenges**: Price skew, missing data (>80% in some columns), outliers.
*   **Final Shape**: 37,813 rows, 66 columns (checksum: `bd5408aef5e09bf9d11b926e5b6fb131`).
*   **EDA Insights** (`notebooks/eda.ipynb`):
    *   Right-skewed prices (log-transformed).
    *   Negative correlation: `km`/`car_age` vs. `listed_price`.
    *   Petrol cars dominant (visualized in `docs/reports/fuel_distribution.png`).
    *   Visuals moved to `docs/reports/`: `price_distribution.png`, `km_vs_price.png`, `car_age_vs_price.png`, `price_by_city.png`, `missing_heatmap.png`.

🔧 Data Processing Pipeline
---------------------------

1.  **Exploratory Data Analysis** (`notebooks/eda.ipynb`):
    *   Computed `car_age` (2025 - `myear`).
    *   Generated 6 visuals and missing values summary (`docs/reports/eda_metadata.json`).
    *   Dataset checksum: `d2c9aa9c030052df712388d301382f49`.
2.  **Cleaning** (`notebooks/cleaning_feature_engineering.ipynb`):
    *   Dropped columns with >80% missing: `ground clearance unladen`, `stroke`.
    *   Imputed medians for numericals (e.g., `alloy wheel size`\=16).
    *   Standardized categoricals with "Unknown" for NaN.
    *   Clipped outliers: `km` (3,358–191,783), `listed_price` (₹84,696–₹4,872,000).
    *   Added `km_per_year` = `km` / `car_age`.
    *   Removed 0 invalid prices.
3.  **Feature Engineering** (`notebooks/feature_engineering_experiments.ipynb`):
    *   Target encoding: `oem_target_enc`, `model_target_enc`, `city_target_enc`.
    *   Frequency encoding for `oem`, `model`, `city`.
    *   Interactions: `brand_age` (`car_age` \* `oem_target_enc`), `km_per_year_age`, `power_weight_ratio`.

🤖 Model Training
-----------------

*   **Baseline Model** (`notebooks/baseline_model.ipynb`):
    *   LightGBM Regressor on 22 numerical + 14 categorical features, log-transformed `listed_price`.
    *   80/20 train-test split.
    *   Performance: RMSE ₹143,383, MAE ₹72,222, R² 0.9622.
    *   Saved: `models/baseline_model.pkl`, `models/scaler.pkl`, `models/feature_names.pkl`.
*   **Improved Model** (`notebooks/feature_engineering_experiments.ipynb`):
    *   Added encodings and interactions.
    *   Performance: RMSE ₹140,458, MAE ₹73,241, R² 0.9686.
*   **RandomForest Comparison** (`notebooks/anomaly_scoring.ipynb`):
    *   Performance: RMSE ₹162,694, R² 0.9579.
    *   Top features: `width` (0.416), `myear` (0.224).

🔍 Anomaly Detection
--------------------

*   **Residuals Analysis** (`notebooks/anomaly_scoring.ipynb`):
    *   Flagged 1,217 anomalies (residuals >2 standard deviations, threshold ₹257,501).
*   **SHAP Explainability** (`anomaly_detector/shap_explainability.py`):
    *   Explained LightGBM predictions with SHAP values.
    *   Visuals moved to `docs/reports/`: `shap_summary_plot.png`, `shap_dependence_width.png`, `shap_force_plot_anomaly_1259.png`, `predicted_vs_actual.png`.
    *   Anomaly data saved: `reports/anomalies.csv`.
*   **Business Rules Validation** (`anomaly_detector/business_rules.py`):
    *   Applied rules: high price for high mileage/old cars, high SHAP contributions for `width`/`myear`.
    *   Validated **103 anomalies**: 96 `high_price_high_km`, 6 `high_price_old_car`, 1 combined.
    *   Bar plot moved to `docs/reports/rule_violations_bar.png`.
    *   Validated anomalies saved: `reports/validated_anomalies.csv`.

📱 Interactive Streamlit App
----------------------------

*   **Implementation** (`anomaly_detector/streamlit_demo_ui.py`):
    *   Deployed at [price-anomaly-detector.streamlit.app](https://price-anomaly-detector.streamlit.app).
    *   **Features**:
        *   Multiselect filters: Rule violations (`high_price_high_km`, `high_price_old_car`), OEM (e.g., Maruti, Hyundai), Model (e.g., Swift, Dzire).
        *   Dynamic filter options: OEM/models update based on prior selections.
        *   Sortable table displaying 103 anomalies (columns: `model`, `oem`, `listed_price`, `predicted_price`, `residual`, `km`, `car_age`, `rule_violations`).
        *   Visuals: Scatter plot (predicted vs. actual prices, `Rs.` formatted, rotated x-axis labels), pie chart (rule violations: ~93% `high_price_high_km`), SHAP summary/force plots.
        *   Report generation: PDF/CSV with anomaly details, rule counts, visuals.
    *   **Enhancements**:
        *   Fixed scatter plot to reflect filters.
        *   Used `Rs.` instead of `₹` to avoid PDF font issues.
        *   Rotated x-axis labels (45°) in scatter plots (UI and PDF) with `MaxNLocator` for readability.
        *   Resolved Arrow serialization for DataFrame display.
*   **Report Generation** (`anomaly_detector/automated_report_generation.py`):
    *   Generates PDF (`reports/anomaly_report.pdf`) with summary (103 anomalies), table, scatter plot (x-axis labels rotated 45°), SHAP summary.
    *   CSV outputs: `reports/anomaly_report.csv`, `reports/anomaly_report_summary.csv`.

📊 Evaluation
-------------

*   **Holdout Set** (`anomaly_detector/evaluation_holdout.py`):
    *   Evaluated on 20% holdout with 378 synthetic anomalies.
    *   **Metrics**:
        *   LightGBM: Precision 0.6723, Recall 0.3148, F1 0.4288.
        *   RandomForest: Precision 0.6321, Recall 0.3228, F1 0.4273.
    *   Results saved: `docs/reports/evaluation_results.csv`.
    *   Precision-recall plot: `docs/reports/precision_recall_comparison.png`.

🚀 Packaging & Deployment
-------------------------

*   **Packaging** (`anomaly_detector/`):
    *   Created `anomaly_detector` module with `main.py` entry point.
    *   Generated `requirements.txt` (pinned: `pandas==2.2.2`, `lightgbm==4.3.0`, etc.).
    *   Documented setup in `docs/README.md`.
*   **Deployment**:
    *   Deployed to Streamlit Cloud on 2025-09-18.
    *   Used Git LFS for large files: `data/cleaned_engineered.csv` (74.89 MB), `cars_data_clean.csv` (73.49 MB), `cleaned_with_age.csv` (73.50 MB), `models/*.pkl`.
    *   Configured `Procfile`, `setup.sh`, relative paths.
    *   GitHub: [shreyyyjain/price-anomaly-detector](https://github.com/shreyyyjain/price-anomaly-detector).
*   **Repository Cleanup**:
    *   Removed redundant files: `.ipynb_checkpoints/`, `catboost_info/`, `Main.ipynb`, `Setup.ipynb`, `app.py`, `automated_report_generation.py`, `deploy_streamlit.py`, `evaluation_holdout.py`, `packaging_docs.py`, `scaler.pkl`, `streamlit_demo_ui.py`, `top_50_anomalies.csv`.
    *   Moved non-essential reports (e.g., `.png`, `.csv`) to `docs/reports/`.
    *   Kept: `anomaly_detector/`, `data/`, `docs/`, `models/`, `notebooks/`, `reports/validated_anomalies.csv`, `.gitattributes`, `.gitignore`, `Procfile`, `README.md`, `requirements.txt`, `setup.sh`.

🔮 Future Work
--------------

*   Add rules (e.g., low price for premium brands).
*   Integrate real-time CarDekho data.
*   Develop a mobile app.
*   Improve recall (0.31-0.32) via dynamic thresholding.

🛠️ Setup Instructions
----------------------

1.  Clone the repository:
    
        git clone https://github.com/shreyyyjain/price-anomaly-detector.git
        cd price-anomaly-detector
        
    
2.  Install Git LFS and pull large files:
    
        git lfs install
        git lfs pull
        
    
3.  Set up virtual environment and install dependencies:
    
        python -m venv .venv
        source .venv/bin/activate  # Windows: .venv\Scripts\activate
        pip install -r requirements.txt
        
    
4.  Run the Streamlit app locally:
    
        streamlit run anomaly_detector/main.py
        
    
5.  Access the deployed app: [price-anomaly-detector.streamlit.app](https://price-anomaly-detector.streamlit.app).