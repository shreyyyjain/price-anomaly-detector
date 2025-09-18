
# EDA Summary
- Dataset: CarDekho used cars (37813 rows, 67 columns).
- Key features: listed_price, km, myear, car_age (computed), fuel, transmission, city.
- Price is right-skewed; log transformation recommended.
- Visuals saved in reports/: price distribution, km vs. price, car_age vs. price, fuel, transmission, and price by city.
- Missing values summary stored in reports/eda_metadata.json and heatmap.
- Dataset MD5 checksum: d2c9aa9c030052df712388d301382f49
- Next steps: Clean missing values, encode categoricals, and train baseline model.

# Cleaning and Feature Engineering Summary
- Started with dataset: 37813 rows.
- Dropped columns with >80% missing values: ['ground clearance unladen', 'stroke'].
- Imputed numerical columns with medians: ['km', 'car_age', 'max power delivered', 'alloy wheel size', 'wheel base', 'no of cylinder', 'length', 'width', 'height', 'top speed', 'acceleration', 'kerb weight', 'gross weight', 'front tread', 'rear tread', 'turning radius', 'cargo volume', 'max torque delivered', 'max power at', 'max torque at', 'bore'].
- Standardized categorical columns, filled NaN with 'Unknown'.
- Clipped outliers in km and listed_price (1st/99th percentiles).
- Added feature: km_per_year = km / car_age.
- Removed rows with non-positive prices.
- Final dataset shape: (37813, 66), checksum: bd5408aef5e09bf9d11b926e5b6fb131.
- Next steps: Train baseline model for price prediction.

# Optimized Baseline Model Summary
- Trained LightGBM Regressor with 22 numerical and 14 categorical features.
- Log-transformed target (listed_price) due to skew.
- Train/test split: 80/20.
- Performance (log scale): RMSE=0.1576, MAE=0.1094, R²=0.9622.
- Performance (actual price, ₹): RMSE=143,383, MAE=72,222.
- Model and scaler saved to models/baseline_model.pkl and models/scaler.pkl.
- Feature names saved to models/feature_names.pkl.
- Next steps: Feature engineering, SHAP-based anomaly scoring.

# SHAP Explainability Summary
- Used SHAP to explain LightGBM model predictions.
- Identified 1217 anomalies based on residuals > 2 standard deviations (threshold: ₹257,501).
- Generated SHAP summary plot, dependence plot for 'width', force plot for a sample anomaly, and predicted vs. actual scatter plot.
- Visuals saved in reports/: shap_summary_plot.png, shap_dependence_width.png, shap_force_plot_anomaly_1259.png, predicted_vs_actual.png.
- Anomaly data saved to reports/anomalies.csv.
- Next steps: Implement business logic and rule-based checks for anomaly validation.

# Business Rules Summary
- Applied business rules to validate 103 out of 1217 anomalies from SHAP analysis.
- Rules include: low price for low mileage, low price for recent cars, high price for old cars, high price for high mileage, and high SHAP contributions for 'width' or 'myear'.
- Rule violation counts: {'high_price_high_km': 97, 'high_price_old_car': 7}.
- Bar plot of rule violations saved to reports/rule_violations_bar.png.
- Validated anomalies saved to reports/validated_anomalies.csv.
- Next steps: Build Streamlit demo skeleton for interactive visualization.

# Streamlit Demo Skeleton Summary
- Built a Streamlit app to display 103 validated anomalies from business rules.
- Features: Filter by rule violations (high_price_high_km, high_price_old_car), view anomaly details, and display SHAP force plots.
- Resolved Arrow serialization issues for DataFrame display.
- Next steps: Polish UI with additional visuals and interactive features.

# Evaluation Summary
- Evaluated anomaly detection on a holdout set with 378 synthetic anomalies.
- Metrics:
  - LightGBM: Precision=0.6723, Recall=0.3148, F1=0.4288
  - RandomForest: Precision=0.6321, Recall=0.3228, F1=0.4273
- Results saved to reports/evaluation_results.csv.
- Precision-recall comparison plot saved to reports/precision_recall_comparison.png.
- Next steps: Package the project and prepare documentation.

# Streamlit UI Summary
- Enhanced Streamlit app with 103 validated anomalies.
- Added scatter plot, SHAP summary plot for anomalies, sorting, and dynamic rule filtering.
- Fixed SHAP force plot visualization issues.
- Resolved Arrow serialization issues for DataFrame display.
- Next steps: Add automated report generation for anomalies.

# Project Summary
- Completed packaging and documentation on 2025-09-18 16:50:25.
- Created anomaly_detector module with main.py entry point.
- Generated requirements.txt and detailed documentation in docs/README.md.
- Next steps: Deploy Streamlit app to a cloud platform.

# Evaluation Summary
- Evaluated anomaly detection on a holdout set with 378 synthetic anomalies.
- Metrics:
  - LightGBM: Precision=0.6448, Recall=0.3122, F1=0.4207
  - RandomForest: Precision=0.6321, Recall=0.3228, F1=0.4273
- Results saved to reports/evaluation_results.csv.
- Precision-recall comparison plot saved to reports/precision_recall_comparison.png.
- Next steps: Package the project and prepare documentation.
