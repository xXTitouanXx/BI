from fastapi import FastAPI, HTTPException, File, UploadFile, Request
import pickle
import pandas as pd
import os

ROOT_DIR = os.path.dirname(os.getcwd())

DATA_DIR = os.path.join(ROOT_DIR, 'data')
PKL_DIR = os.path.join(ROOT_DIR, 'artifacts') 

PROD_DATA_PATH = os.path.join(DATA_DIR, 'prod_data.csv')
EMBEDDING_PATH = os.path.join(PKL_DIR, 'embedding.pkl')
PIPELINE_PATH = os.path.join(PKL_DIR, 'pipeline.pkl')

with open(EMBEDDING_PATH, "rb") as f:
    EMBEDDING = pickle.load(f)

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
    
    data_df = pd.DataFrame([data], columns=EXPECTED_COLUMNS)    
    data_df["Target"] = true_target
    data_df["Prediction"] = prediction
    
    if not os.path.exists(PROD_DATA_PATH):
        data_df.to_csv(PROD_DATA_PATH, index=False)
    else:
        data_df.to_csv(PROD_DATA_PATH, mode="a", header=False, index=False)
    
    return {"message": "Feedback enregistré avec succès !"}