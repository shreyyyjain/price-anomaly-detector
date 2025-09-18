import os
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Project paths (relative for deployment)
DATA_PATH = 'data/cleaned_engineered.csv'
ANOMALIES_PATH = 'reports/validated_anomalies.csv'
MODEL_PATH = 'models/baseline_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURE_PATH = 'models/feature_names.pkl'
REPORTS_PATH = 'reports'
REPORT_PATH = os.path.join(REPORTS_PATH, 'anomaly_report.pdf')

os.makedirs(REPORTS_PATH, exist_ok=True)

def generate_report(filtered_anomalies, X_anomalies, shap_values, explainer):
    # [Previous generate_report function unchanged]
    # Create PDF document
    doc = SimpleDocTemplate(REPORT_PATH, pagesize=letter, leftMargin=36, rightMargin=36)
    elements = []

    # Title
    styles = getSampleStyleSheet()
    title = Paragraph("Anomaly Detection Report", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    elements.append(PageBreak())

    # Summary with rule violation counts
    rule_counts = {}
    for violations in filtered_anomalies['rule_violations'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [x]):
        for rule in violations:
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
    summary = f"Total Anomalies: {len(filtered_anomalies)}<br/>Generated: {pd.Timestamp.now()}<br/>Rule Violations: {rule_counts}"
    elements.append(Paragraph("Summary", styles['Heading2']))
    elements.append(Paragraph(summary, styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(PageBreak())

    # Anomaly Table
    display_cols = ['model', 'oem', 'listed_price', 'predicted_price', 'residual', 'km', 'car_age', 'rule_violations']
    report_df = filtered_anomalies[display_cols].copy()
    for col in ['listed_price', 'predicted_price', 'residual']:
        report_df[col] = report_df[col].apply(lambda x: f'₹{x:,.0f}' if pd.notnull(x) else '₹0')
    for col in ['km', 'car_age']:
        report_df[col] = report_df[col].apply(lambda x: f'{x:,.0f}' if pd.notnull(x) else '0')
    
    normal_style = styles['Normal']
    table_data = [display_cols]
    for _, row in report_df.iterrows():
        row_data = []
        for col in display_cols:
            value = str(row[col])
            if col in ['model', 'rule_violations']:
                row_data.append(Paragraph(value, normal_style))
            else:
                row_data.append(value)
        table_data.append(row_data)
    
    col_widths = [100, 80, 70, 70, 70, 50, 50, 100]
    table = Table(table_data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(Paragraph("Anomaly Details", styles['Heading2']))
    elements.append(table)
    elements.append(Spacer(1, 12))
    elements.append(PageBreak())

    # Scatter Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(filtered_anomalies['listed_price'], filtered_anomalies['predicted_price'], c='red', alpha=0.5, label='Anomaly')
    ax.plot([filtered_anomalies['listed_price'].min(), filtered_anomalies['listed_price'].max()],
            [filtered_anomalies['listed_price'].min(), filtered_anomalies['listed_price'].max()], 'k--')
    ax.set_xlabel('Actual Price (₹)')
    ax.set_ylabel('Predicted Price (₹)')
    ax.set_title('Predicted vs. Actual Prices')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    img_path = os.path.join(REPORTS_PATH, 'scatter_plot.png')
    fig.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    elements.append(Paragraph("Predicted vs. Actual Prices", styles['Heading2']))
    elements.append(Image(img_path, width=350, height=250))
    elements.append(Spacer(1, 12))
    elements.append(PageBreak())

    # SHAP Summary Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    shap.summary_plot(shap_values, X_anomalies, show=False)
    ax.set_title('SHAP Summary for Anomalies')
    img_path = os.path.join(REPORTS_PATH, 'shap_summary.png')
    fig.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    elements.append(Paragraph("SHAP Summary Plot", styles['Heading2']))
    elements.append(Image(img_path, width=350, height=250))
    elements.append(Spacer(1, 12))

    # Build PDF
    doc.build(elements)

    # Save to CSV
    csv_path = os.path.join(REPORTS_PATH, 'anomaly_report.csv')
    report_df.to_csv(csv_path, index=False)

    # Save summary CSV
    summary_df = pd.DataFrame({
        'Metric': ['Total Anomalies'] + [f'Rule_{rule}' for rule in rule_counts],
        'Value': [len(filtered_anomalies)] + list(rule_counts.values())
    })
    summary_df.to_csv(os.path.join(REPORTS_PATH, 'anomaly_report_summary.csv'), index=False)

    return REPORT_PATH, csv_path

if __name__ == "__main__":
    try:
        # Load data for report generation
        anomaly_df = pd.read_csv(ANOMALIES_PATH, low_memory=False)
        anomaly_df['rule_violations'] = anomaly_df['rule_violations'].apply(lambda x: str(x))
        
        df = pd.read_csv(DATA_PATH, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
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

        X = df[num_cols + cat_cols].copy()
        for col in cat_cols:
            X[col] = X[col].astype('category')
        X[num_cols] = scaler.transform(X[num_cols])

        # Handle missing original_index
        if 'original_index' in anomaly_df.columns:
            anomaly_indices = anomaly_df['original_index'].tolist()
        else:
            print("Warning: 'original_index' column not found in validated_anomalies.csv. Using DataFrame index.")
            anomaly_indices = anomaly_df.index.tolist()
        
        valid_indices = [idx for idx in anomaly_indices if idx in X.index]
        if len(valid_indices) < len(anomaly_indices):
            print(f"Warning: {len(anomaly_indices) - len(valid_indices)} invalid indices found. Using {len(valid_indices)} valid indices.")
        X_anomalies = X.loc[valid_indices]
        if X_anomalies.empty:
            raise ValueError("No valid anomalies found after index filtering.")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_anomalies)

        # Generate report with all anomalies
        filtered_anomalies = anomaly_df
        pdf_path, csv_path = generate_report(filtered_anomalies, X_anomalies, shap_values, explainer)
        print(f"Report generated: PDF at {pdf_path}, CSV at {csv_path}")

    except Exception as e:
        print(f"Error generating report: {e}")