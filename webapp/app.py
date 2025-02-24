import streamlit as st
import requests

st.title("Interface de prédiction")
data = st.file_uploader("Uploader vos données")

if st.button("Prédire"):
    response = requests.post("http://serving-api:8080/predict", json={"data": data.read().decode()})
    st.write(f"Prédiction: {response.json()['prediction']}")