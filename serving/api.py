from fastapi import FastAPI, HTTPException, File, UploadFile
import pickle
import pandas as pd
import os

PKL_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
MODEL_PATH = os.path.join(PKL_DIR, 'best_model.pkl')

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

EXPECTED_COLUMNS = [
    "Age",
    "Pregnancies",
    "BMI",
    "Glucose",
    "BloodPressure",
    "HbA1c",
    "LDL",
    "HDL",
    "Triglycerides",
    "WaistCircumference",
    "HipCircumference",
    "WHR",
    "FamilyHistory",
    "DietType",
    "Hypertension",
    "MedicationUse"
]

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint pour effectuer une prédiction à partir des données fournies en JSON.
    """

    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé, vérifiez les fichiers dans artifacts/")

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la lecture du fichier CSV: {str(e)}")

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes manquantes dans le fichier CSV: {missing_columns}"
        )

    selected_data = df[EXPECTED_COLUMNS]
    input_array = selected_data.to_numpy()
    
    try:
        predictions = MODEL.predict(input_array)
        return {"predictions": predictions.tolist()}
    except:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {e}")
