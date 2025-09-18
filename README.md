# Explainable Price Anomaly Detector for Indian Second-hand Car Marketplace

## 📊 Project Overview
This project builds an explainable ML system to detect over/underpriced used cars in the Indian second-hand market using the CarDekho dataset (37,813 listings). It identifies 103 anomalous listings (96 overpriced for high mileage, 7 for old cars) via LightGBM predictions, SHAP explanations, and business rules. The interactive Streamlit app (deployed at [price-anomaly-detector.streamlit.app](https://price-anomaly-detector.streamlit.app)) enables filtering, visualization, and PDF reports. Key metrics: LightGBM RMSE ₹140,458, R² 0.9686; anomaly precision 0.6723.

**Business Impact**: Helps buyers avoid overpaying (e.g., 96 high-mileage cars priced ₹1M+) and sellers price fairly, building trust in marketplaces like CarDekho.

**Tech Stack**: Python, Pandas, LightGBM, SHAP, Streamlit, ReportLab.

## 📈 Dataset
- **Source**: CarDekho used cars (India).
- **Size**: 37,813 rows, 67 columns.
- **Key Features**: `listed_price` (mean ₹799,986, clipped ₹84,696–₹4,872,000), `km` (mean 62,409, clipped 3,358–191,783), `myear` (car_age mean 9.43 years), `fuel` (petrol dominant), `transmission`, `oem`/`model` (high cardinality).
- **Challenges**: Right-skewed prices, 80% missing columns (e.g., `stroke`), outliers.
- **Preprocessed**: 37,813 rows, 66 columns (checksum: bd5408aef5e09bf9d11b926e5b6fb131). Visuals in `reports/`: price distribution (log-scale skew), km/age vs. price scatters, fuel countplot.

## 🔧 Data Processing Pipeline
1. **EDA** (`notebooks/eda.ipynb`): Computed `car_age`, generated 6 visuals (e.g., missing heatmap, MD5 d2c9aa9c030052df712388d301382f49).
2. **Cleaning** (`notebooks/cleaning_feature_engineering.ipynb`): Dropped >80% missing columns (`ground clearance unladen`, `stroke`), imputed medians (e.g., `alloy wheel size`=16), standardized categoricals ("Unknown"), clipped outliers, added `km_per_year`.
3. **Feature Engineering** (`notebooks/feature_engineering_experiments.ipynb`): Target/frequency encoding (`oem_target_enc`, `model_target_enc`, `city_target_enc`), interactions (`brand_age`, `km_per_year_age`, `power_weight_ratio`).

## 🤖 Model Training
- **Baseline** (`notebooks/baseline_model.ipynb`): LightGBM on 22 numerical + 14 categorical features, log `listed_price`. Test: RMSE ₹143,383, MAE ₹72,222, R² 0.9622.
- **Improved** (`notebooks/feature_engineering_experiments.ipynb`): Enhanced with encodings/interactions. Test: RMSE ₹140,458, MAE ₹73,241, R² 0.9686.
- **RandomForest Comparison** (`notebooks/anomaly_scoring.ipynb`): RMSE ₹162,694, R² 0.9579 (top features: `width` 0.416, `myear` 0.224).

## 🔍 Anomaly Detection
- **Residuals** (`notebooks/anomaly_scoring.ipynb`): Flagged 1,217 anomalies (residuals >2 std).
- **SHAP Explainability** (`shap_explainability.py`): Visuals in `reports/` (summary plot, dependence for `width`, force plot for anomaly 1259, predicted vs. actual scatter). Threshold: ₹257,501.
- **Business Rules** (`business_rules.py`): Validated 103 anomalies (96 `high_price_high_km`, 6 `high_price_old_car`, 1 combined). Bar plot in `reports/rule_violations_bar.png`.

## 📱 Interactive Demo
- **Streamlit App** (`anomaly_detector/streamlit_demo_ui.py`): Deployed at [price-anomaly-detector.streamlit.app](https://price-anomaly-detector.streamlit.app).
  - **Features**: Multiselect filters (rules, OEM, model), sortable table, dynamic scatter/pie charts, SHAP force plots, PDF/CSV reports.
  - **UI Polish**: Rs. formatting, rotated x-axis labels, MaxNLocator for readability.
- **Report Generation** (`automated_report