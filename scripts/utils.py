import pandas as pd
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier






def readfile(file, sep=','):
    df = pd.read_csv(file, sep=sep)

    X = df.drop(columns=['Outcome'])

    y = df['Outcome']

    return X, y


def preprocess(df):
    df = df.copy()  # Éviter de modifier le DataFrame d'origine

    # Suppression des valeurs manquantes
    df.dropna(inplace=True)  # Option : Imputation avec df.fillna(df.mean())
    
    return df
# Fonction pour sauvegarder le meilleur modèle
def save_best_model(model, filename='artifacts/best_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Fonction pour charger le modèle sauvegardé
def load_model(filename='artifacts/best_model.pkl'):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def run_classifiers(models, strategies, X, y):
    best_score = 0
    best_model = None
    best_pipeline = None
    
    for strat_name, strat in strategies.items():
        for model_name, model in models.items():
            pipeline = Pipeline([
                ('preprocessing', strat),
                ('classifier', model)
            ])
            
            scores = cross_val_score(pipeline, X, y, cv=10, scoring='f1_macro')
            mean_score = scores.mean()
            
            print(f"{model_name} + {strat_name} -> F1-moyen : {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = f"{model_name} + {strat_name}"
                # Fit the best pipeline on full dataset
                pipeline.fit(X, y)
                best_pipeline = pipeline
    
    print(f"Meilleur modèle : {best_model}")
    save_best_model(best_pipeline)  # Save the actual fitted pipeline