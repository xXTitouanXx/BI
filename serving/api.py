from fastapi import FastAPI, HTTPException, File, UploadFile, Request
import pickle
import pandas as pd
import os

ROOT_DIR = os.path.dirname(os.getcwd())
PKL_DIR = os.path.join(ROOT_DIR, 'artifacts')   
EMBEDDING_PATH = os.path.join(PKL_DIR, 'embedding.pkl')
SCALER_PATH = os.path.join(PKL_DIR, 'scaler.pkl')
PIPELINE_PATH = os.path.join(PKL_DIR, 'pipeline.pkl')

with open(EMBEDDING_PATH, "rb") as f:
    EMBEDDING = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    SCALER = pickle.load(f)

with open(PIPELINE_PATH, "rb") as f:
    PIPELINE = pickle.load(f)

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

    if PIPELINE is None:
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
        predictions = PIPELINE.predict(input_array)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {e}")

@app.post("/feedback")
async def feedback(request: Request):
    """
    Endpoint pour recevoir les feedbacks et enregistrer les données dans prod_data.csv.
    """
    feedback_data = await request.json()
    
    data = feedback_data["data"]
    prediction = feedback_data["prediction"]
    true_target = feedback_data["true_target"]
    
    data_df = pd.DataFrame([data])

    if SCALER is not None:
        data_scaled = SCALER.transform(data_df)
        data_df = pd.DataFrame(data_scaled, columns=EXPECTED_COLUMNS)
    
    data_df["target"] = true_target
    data_df["prediction"] = prediction
    data_df.to_csv("prod_data.csv", mode="a", header=not os.path.exists("prod_data.csv"), index=False)
    
    return {"message": "Feedback enregistré avec succès !"}