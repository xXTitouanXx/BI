from utils import *

# Dataset source : https://www.kaggle.com/datasets/asinow/diabetes-dataset

# PROD (Mettre à True pour tester le modèle en production sur prod_data.csv)
PROD = False

# Chemins des fichiers de données
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
ref_data_path = os.path.join(data_dir, 'ref_data.csv')
prod_data_path = os.path.join(data_dir, 'prod_data.csv')

# Import des données 
X, y = readfile(ref_data_path)

# Prétraitement des données
X = preprocess(X)

# Détection des colonnes numériques et catégorielles
num_cols = ['Age', 'Pregnancies', 'BMI', 'Glucose', 'BloodPressure', 'HbA1c',
            'LDL', 'HDL', 'Triglycerides', 'WaistCircumference', 'HipCircumference', 'WHR']

cat_cols = ['FamilyHistory', 'DietType', 'Hypertension', 'MedicationUse']  # Variables binaires

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', 'passthrough', cat_cols)  # Les catégories sont déjà encodées
    ])

# Stratégies de prétraitement
strategies = {
    "Raw": None,
    "Standardized": preprocessor,
    "PCA": Pipeline([
        ('preproc', preprocessor),
        ('pca', PCA(n_components=0.95))
    ])
}

# Modèles à tester
models = {
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=1),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=1),
    'MLP': MLPClassifier(hidden_layer_sizes=(20,10), max_iter=1000, random_state=1),
    'k-NN': KNeighborsClassifier(n_neighbors=5),
    'Bagging': BaggingClassifier(n_estimators=200, random_state=1)
}
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

run_classifiers(models, strategies, X_resampled, y_resampled)

### TEST DE PRODUCTION ###
if PROD:
    
    # Charger le modèle sauvegardé
    best_pipeline = load_model()  

    # Lire les données de production
    prod_data = pd.read_csv(prod_data_path)
    X_prod = prod_data.drop(columns=['Outcome'])
    y_prod = prod_data['Outcome']

    # Prédictions
    y_pred = best_pipeline.predict(X_prod)

    # Évaluer le taux de prédictions correctes
    accuracy = accuracy_score(y_prod, y_pred)
    print(f"Taux de prédictions correctes : {accuracy:.4f}")