
# Deployment Instructions for Streamlit Cloud

1. **Initialize a Git Repository** (if not already done):
   ```bash
   cd "C:\Users\shrey\Desktop\Projects\Explainable Price Anomaly Detector for Indian Second-hand Marketplace"
   git init
   git add .
   git commit -m "Initial commit for Price Anomaly Detector"
   ```

2. **Create a GitHub Repository**:
   - Go to https://github.com/new and create a repository (e.g., `price-anomaly-detector`).
   - Push your project to GitHub:
     ```bash
     git remote add origin https://github.com/your-username/price-anomaly-detector.git
     git branch -M main
     git push -u origin main
     ```

3. **Deploy to Streamlit Cloud**:
   - Go to https://streamlit.io/cloud and sign in with GitHub.
   - Click "New app" > "From existing repo".
   - Select your repository (`your-username/price-anomaly-detector`).
   - Set the main file path to `anomaly_detector/main.py`.
   - Click "Deploy" and wait for the app to build.

4. **Verify Deployment**:
   - Once deployed, access the app at the provided URL (e.g., https://your-app-name.streamlit.app).
   - Test the UI, filtering, and report generation.

5. **Troubleshooting**:
   - Check Streamlit Cloud logs for errors.
   - Ensure all dependencies in requirements.txt are compatible.
   - Verify that `data/`, `models/`, and `reports/` directories are excluded in .gitignore (they should be uploaded separately if needed).

**Note**: If you include `data/`, `models/`, or `reports/` in the repository, ensure `cleaned_engineered.csv`, `baseline_model.pkl`, `scaler.pkl`, `feature_names.pkl`, and `validated_anomalies.csv` are present. Alternatively, configure the app to download these from a cloud storage service (e.g., Google Drive, AWS S3).
