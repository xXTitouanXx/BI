import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
REF_DATA_PATH = os.path.join(DATA_DIR, 'ref_data.csv')
PROD_DATA_PATH = os.path.join(DATA_DIR, 'prod_data.csv')
REPORT_DATA_PATH = os.path.join(DATA_DIR, 'report.html')

ref_data = pd.read_csv(REF_DATA_PATH, sep=',')
prod_data = pd.read_csv(PROD_DATA_PATH, sep=',')

if "Target" in prod_data.columns:
    prod_data = prod_data.rename(columns={"Target": "Outcome"})

if "Prediction" in prod_data.columns:
    prod_data = prod_data.drop(columns=["Prediction"])

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_data, current_data=prod_data)
report.save_html(REPORT_DATA_PATH)
