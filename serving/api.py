from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
import pandas as pd
import os

# Initialisation de l'API
app = FastAPI()

PKL_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
MODEL_PATH = os.path.join(PKL_DIR, 'best_model.pkl')
MODEL = joblib.load(MODEL_PATH)

# Définition du format d'entrée attendu avec Pydantic
class InputData(BaseModel):
    Age: int
    Pregnancies: int
    BMI: float
    Glucose: float
    BloodPressure: float
    HbA1c: float
    LDL: float
    HDL: float
    Triglycerides: float
    WaistCircumference: float
    HipCircumference: float
    WHR: float
    FamilyHistory: int
    DietType: int
    Hypertension: int
    MedicationUse: int


@app.post("/predict")
async def predict(data: InputData):
    """
    Endpoint pour effectuer une prédiction à partir des données fournies en JSON.
    """

    # Vérification que le modèle est bien chargé
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé, vérifiez les fichiers dans artifacts/")

    try:
        # Conversion en DataFrame
        df = pd.DataFrame([data.dict()])

        # Prédiction avec le modèle
        prediction = MODEL.predict(df).tolist()

        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {e}")


@app.post("/feedback")
async def feedback(data: dict):
    # Ajouter les données à prod_data.csv
    # Déclencher le ré-entraînement si nécessaire
    return {"status": "success"}