import streamlit as st
import requests
import pandas as pd

if "selected_index" not in st.session_state:
    st.session_state.selected_index = 0
if "true_target" not in st.session_state:
    st.session_state.true_target = 0
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "df_preview" not in st.session_state:
    st.session_state.df_preview = None

st.title("Interface de prédiction")
data = st.file_uploader("Uploader vos données", type=["csv"])

if data is not None and data.name.split(".")[-1] == "csv":
    st.session_state.df_preview = pd.read_csv(data)
    st.write("Aperçu des données :", st.session_state.df_preview.head())
    data.seek(0)

if st.button("Prédire"):
    if data is not None:
        data.seek(0)
        response = requests.post("http://serving-api:8080/predict", files={"file": ("data.csv", data, "text/csv")})
        if response.status_code == 200:
            st.session_state.predictions = response.json()["predictions"]
            st.write("Prédictions :", st.session_state.predictions)
        else:
            st.error(f"Erreur lors de la prédiction : {response.status_code} - {response.text}")
    else:
        st.warning("Veuillez uploader un fichier CSV avant de cliquer sur 'Prédire'.")

if st.session_state.predictions:
    st.subheader("Feedback")
    st.session_state.selected_index = st.selectbox(
        "Sélectionnez une prédiction pour fournir un feedback :",
        range(len(st.session_state.predictions)),
        index=st.session_state.selected_index
    )
    st.session_state.true_target = st.number_input(
        "Entrez la cible réelle (valeur attendue) :",
        step=1,
        value=st.session_state.true_target
    )
    
    if st.button("Envoyer Feedback"):
        if st.session_state.df_preview is not None:
            feedback_data = {
                "data": st.session_state.df_preview.iloc[st.session_state.selected_index].to_dict(),
                "prediction": st.session_state.predictions[st.session_state.selected_index],
                "true_target": st.session_state.true_target
            }
            feedback_response = requests.post("http://serving-api:8080/feedback", json=feedback_data)
            if feedback_response.status_code == 200:
                st.success("Feedback envoyé avec succès !")
            else:
                st.error(f"Erreur lors de l'envoi du feedback : {feedback_response.text}")
        else:
            st.error("Aucune donnée disponible pour le feedback.")