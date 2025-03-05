import streamlit as st
import requests
import pandas as pd

st.title("Interface de prédiction")
data = st.file_uploader("Uploader vos données", type=["csv"])

if data is not None and data.name.split(".")[-1] == "csv":
    df_preview = pd.read_csv(data)
    st.write("Aperçu des données :", df_preview.head())
    data.seek(0)

if st.button("Prédire"):
    if data is not None:
        data.seek(0)
        response = requests.post("http://serving-api:8080/predict", files={"file": ("data.csv", data, "text/csv")})
        if response.status_code == 200:
            predictions = response.json()["predictions"]
            st.write("Prédictions :", predictions)
            st.subheader("Feedback")
            true_target = st.number_input("Entrez la cible réelle (valeur attendue) :", step=1)
            if st.button("Envoyer Feedback"):
                feedback_data = {
                    "data": df_preview.to_dict(orient="records")[0],
                    "prediction": predictions[0],
                    "true_target": true_target
                }
                feedback_response = requests.post("http://serving-api:8080/feedback", json=feedback_data)
                if feedback_response.status_code == 200:
                    st.success("Feedback envoyé avec succès !")
                else:
                    st.error(f"Erreur lors de l'envoi du feedback : {feedback_response.text}")
        else:
            st.error(f"Erreur lors de la prédiction : {response.status_code} - {response.text}")
    else:
        st.warning("Veuillez uploader un fichier CSV avant de cliquer sur 'Prédire'.")