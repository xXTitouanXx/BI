# serving/api.py
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('/artifacts/model.pkl')

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    return {"prediction": model.predict(df).tolist()}

@app.post("/feedback")
async def feedback(data: dict):
    # Ajouter les données à prod_data.csv
    # Déclencher le ré-entraînement si nécessaire
    return {"status": "success"}