import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ROOT_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(ROOT_DIR, 'data')
REF_DATA_PATH = os.path.join(DATA_DIR, 'ref_data.csv')
PROD_DATA_PATH = os.path.join(DATA_DIR, 'prod_data.csv')

ref_data = pd.read_csv(REF_DATA_PATH)
prod_data = pd.read_csv(PROD_DATA_PATH)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_data, current_data=prod_data)
report.save_html("report.html")