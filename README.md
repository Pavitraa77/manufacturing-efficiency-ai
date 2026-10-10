# Manufacturing Efficiency AI

An educational machine-learning application that classifies manufacturing efficiency as **High**, **Medium**, or **Low** from machine and sensor readings.

## Features
- Interactive Streamlit dashboard and dataset overview
- Random Forest classifier with categorical feature encoding
- Held-out test accuracy, confusion matrix, and classification report
- Interactive prediction form
- CSV upload, exploration, and download

## Important dataset note
`data/manufacturing_demo.csv` is a **synthetic demo dataset generated for this project**. It is not the original Thales Group dataset and must not be described as real factory data. Upload your approved `Thales_Group_Manufacturing.csv` in the app if you have permission to use it.

The app requires these columns:
`Machine_ID`, `Operation_Mode`, `Temperature_C`, `Vibration_Hz`, `Power_Consumption_kW`, `Network_Latency_ms`, `Packet_Loss_%`, `Quality_Control_Defect_Rate_%`, `Predictive_Maintenance_Score`, `Efficiency_Status`.

## Run locally
Python 3.11 or 3.12 recommended.

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository named `manufacturing-efficiency-ai`.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the `data/` folder.
3. Go to https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
5. Click **Deploy** and wait for the build to finish.

## Workflow
1. Validate and load data.
2. Select machine and sensor features.
3. Split data into train/test sets.
4. One-hot encode categorical columns.
5. Train a class-balanced Random Forest.
6. Evaluate on held-out test data.
7. Serve visualizations and predictions through Streamlit.

## Limitations
This project does not connect to live industrial machines or a 6G network. Predictions are educational estimates, not validated for safety-critical production decisions. Results depend on data quality and labels.
