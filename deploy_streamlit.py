import os
import subprocess
import pandas as pd

# Project paths
PROJECT_ROOT = r'C:\Users\shrey\Desktop\Projects\Explainable Price Anomaly Detector for Indian Second-hand Marketplace'
REPORTS_PATH = os.path.join(PROJECT_ROOT, 'reports')
DOCS_PATH = os.path.join(PROJECT_ROOT, 'docs')
MODULE_PATH = os.path.join(PROJECT_ROOT, 'anomaly_detector')

# Update requirements.txt with pinned versions
def update_requirements():
    packages = [
        'pandas==2.2.3',
        'numpy==1.26.4',
        'lightgbm==4.5.0',
        'scikit-learn==1.7.2',
        'matplotlib==3.9.2',
        'shap==0.46.0',
        'streamlit==1.39.0',
        'reportlab==4.2.2'
    ]
    with open(os.path.join(PROJECT_ROOT, 'requirements.txt'), 'w', encoding='utf-8') as f:
        for pkg in packages:
            f.write(f"{pkg}\n")
    print("requirements.txt updated with pinned versions.")

# Create Procfile for Streamlit Cloud
def create_procfile():
    procfile_content = """
web: sh setup.sh && streamlit run anomaly_detector/main.py --server.port $PORT
"""
    with open(os.path.join(PROJECT_ROOT, 'Procfile'), 'w', encoding='utf-8') as f:
        f.write(procfile_content)
    print("Procfile created.")

# Create setup.sh for Streamlit Cloud
def create_setup_sh():
    setup_content = """
#!/bin/bash
# Setup script for Streamlit Cloud
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = \$PORT
enableCORS = false
" > ~/.streamlit/config.toml
"""
    with open(os.path.join(PROJECT_ROOT, 'setup.sh'), 'w', encoding='utf-8') as f:
        f.write(setup_content)
    print("setup.sh created.")

# Create .gitignore
def create_gitignore():
    gitignore_content = """
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
*.egg-info/
dist/
build/

# Data and models
data/
models/
reports/*.csv
reports/*.png
reports/*.pdf

# Notebooks
notebooks/*.ipynb_checkpoints/

# IDE and OS
.vscode/
.idea/
*.DS_Store
"""
    with open(os.path.join(PROJECT_ROOT, '.gitignore'), 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    print(".gitignore created.")

# Deployment instructions
def print_deployment_instructions():
    instructions = """
# Deployment Instructions for Streamlit Cloud

1. **Initialize a Git Repository** (if not already done):
   ```bash
   cd "{PROJECT_ROOT}"
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
"""
    with open(os.path.join(DOCS_PATH, 'deployment_instructions.md'), 'w', encoding='utf-8') as f:
        f.write(instructions.format(PROJECT_ROOT=PROJECT_ROOT))
    print("Deployment instructions saved to docs/deployment_instructions.md.")
    print(instructions.format(PROJECT_ROOT=PROJECT_ROOT))

# Execute tasks
update_requirements()
create_procfile()
create_setup_sh()
create_gitignore()
print_deployment_instructions()