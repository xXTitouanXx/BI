import streamlit as st
import requests
import pandas as pd

st.title("Interface de prédiction")
data = st.file_uploader("Uploader vos données", type=["csv"])

if data is not None and data.name.split(".")[-1] == "csv":
    df_preview = pd.read_csv(data)
    st.write("Aperçu des données :", df_preview.head())

if st.button("Prédire"):
    if data is not None:
        response = requests.post("http://serving-api:8080/predict", files={"file": ("data.csv", data, "text/csv")})
        if response.status_code == 200:
            st.write("Prédictions :", response.json())
        else:
            st.error(f"Erreur lors de la prédiction : {response.status_code} - {response.text}")
    else:
        st.warning("Veuillez uploader un fichier CSV avant de cliquer sur 'Prédire'.")