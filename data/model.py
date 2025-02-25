import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier

# Fonctions : 

def readfile(file, sep=','):
    df = pd.read_csv(file, sep=sep)

    X = df.drop(columns=['Outcome'])

    y = df['Outcome']

    return X, y


def preprocess(df):
    df = df.copy()  # Éviter de modifier le DataFrame d'origine

    # Suppression des valeurs manquantes
    df.dropna(inplace=True)  # Option : Imputation avec df.fillna(df.mean())

    # Détection des colonnes numériques et catégorielles
    num_cols = ['Age', 'Pregnancies', 'BMI', 'Glucose', 'BloodPressure', 'HbA1c',
                'LDL', 'HDL', 'Triglycerides', 'WaistCircumference', 'HipCircumference', 'WHR']
    
    cat_cols = ['FamilyHistory', 'DietType', 'Hypertension', 'MedicationUse']  # Variables binaires
    
    # Suppression des valeurs aberrantes (Basé sur IQR)
    # for col in num_cols:
    #     Q1 = df[col].quantile(0.25)
    #     Q3 = df[col].quantile(0.75)
    #     IQR = Q3 - Q1
    #     lower_bound = Q1 - 1.5 * IQR
    #     upper_bound = Q3 + 1.5 * IQR
    #     df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # Standardisation des colonnes numériques
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Encodage des variables catégorielles 
    for col in cat_cols:
        if df[col].nunique() > 2:
            df = pd.get_dummies(df, columns=[col], drop_first=True)  # One-Hot Encoding
    
    return df

def run_classifiers(models, strategy, X_train, y_train, X_test, y_test):
    best_score = 0
    best_model = None
    best_strategy = None

    for strat_name, strat in strategy.items():
        for model_name, model in models.items():
            steps = []
            if strat:
                steps.append(('preprocessing', strat))
            steps.append(('classifier', model))
            
            pipeline = Pipeline(steps)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # precision_0 = precision_score(y_test, y_pred, pos_label=0)
            # precision_1 = precision_score(y_test, y_pred, pos_label=1)
            # score = 0.5 * (precision_0 + precision_1)

            # Alternative 1 : F1-score macro
            # score = f1_score(y_test, y_pred, average='macro')
            
            # Alternative 2 : Rappel pondéré (si focus sur éviter les FN)
            score = recall_score(y_test, y_pred, average='weighted')



            print(f"{model_name} with {strat_name} -> Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_strategy = strat

    print(f"Best model: {best_model} with {best_strategy}")

# Dataset source : https://www.kaggle.com/datasets/asinow/diabetes-dataset

# Import des données 
X, y = readfile('data/diabetes_dataset.csv')

# Prétraitement des données
X = preprocess(X)
 
# Séparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Modèles à tester
models = {
    'NaiveBayes': GaussianNB(),
    'CART': DecisionTreeClassifier(random_state=1),
    'ID3': DecisionTreeClassifier(criterion='entropy', random_state=1),
    'DecisionStump': DecisionTreeClassifier(max_depth=1, random_state=1),
    'MLP': MLPClassifier(hidden_layer_sizes=(20,10), max_iter=1000, random_state=1),
    'k-NN': KNeighborsClassifier(n_neighbors=5),
    'Bagging': BaggingClassifier(n_estimators=200, random_state=1),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=1),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=1),
    'XGBoost': XGBClassifier(n_estimators=200, random_state=1, use_label_encoder=False)
}

# Stratégies de prétraitement
strategies = {
    "Raw": None,
    "Standardized": StandardScaler(),
    "Standardized + PCA": Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95))
    ])
}

run_classifiers(models, strategies, X_train, y_train, X_test, y_test)