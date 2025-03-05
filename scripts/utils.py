import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import pickle
import time

def load_csv(filename, test_size=0.5):
    dataframe = pd.read_csv(filename)
    data = np.array([row.split(';') for row in dataframe.values.flatten()], dtype=float)

    x = data[:, :-1]
    y = data[:, -1]
    features = np.array(dataframe.columns.values[0].split(';'))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)
    
    print("Taille de l'échantillon:", x.shape)
    print("Pourcentage d'exemples positifs:", np.sum(y == 1) / len(y) * 100)
    print("Pourcentage d'exemples négatifs:", np.sum(y == 0) / len(y) * 100)
    print("Features:", features)
    print("Training set size:", x_train.shape)
    print("Test set size:", x_test.shape)
    
    return x_train, x_test, y_train, y_test, features

def load_data(filename, test_size=0.2, raw=False, split=True):
    data = pd.read_csv(filename, header=None, sep='\t')

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if not raw:
        is_numeric = x.apply(lambda col: pd.to_numeric(col, errors='coerce').notna().all())
        x = np.array(x.loc[:, is_numeric].astype(float))
        x[x == '?'] = np.nan
        x = x.astype(float)

    y = np.where(y == '+', 1, 0)

    x = np.array(x)

    print(f"Sample size: {x.shape}")
    print(f"Positive examples: {np.sum(y == 1)}")
    print(f"Negative examples: {np.sum(y == 0)}")

    if not split:
        return x, y

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)
    print("Training set size:", x_train.shape)
    print("Test set size:", x_test.shape)
    
    return x_train, x_test, y_train, y_test

def train(model, x_train, y_train, x_test):
    model.fit(x_train, y_train.astype(int))
    return model.predict(x_test)

def evaluate(model, y_pred, y_test, verbose=False):
    accuracy = accuracy_score(y_test.astype(int), y_pred)
    precision = precision_score(y_test.astype(int), y_pred, zero_division=1)
    if verbose:
        print(model)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
    return accuracy, precision

def standardise(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled

def linear_combination(x_train, x_test, n_components=3):
    pca = PCA(n_components=n_components)
    x_train_scaled, x_test_scaled = standardise(x_train, x_test)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    x_train_concat = np.concatenate((x_train_scaled, x_train_pca), axis=1)
    x_test_concat = np.concatenate((x_test_scaled, x_test_pca), axis=1)
    return x_train_concat, x_test_concat

def benchmark(models, samples, y_train, y_test, verbose=False, veryverbose=False):
    accuracies = {}
    precisions = {}
    best_model = None
    best_score = 0
    for name, model in models.items():
        accuracies[name] = {}
        precisions[name] = {}
        for method, sample in samples.items():
            x_train, x_test = sample
            y_pred = train(model, x_train, y_train, x_test)
            accuracy, precision = evaluate(model, y_pred, y_test, veryverbose)
            accuracies[name][method] = accuracy
            precisions[name][method] = precision

            avg_score = (accuracy + precision) / 2
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = (name, method)
    if verbose:
        print('Best model:', best_model)
    return accuracies, precisions, best_model

def random_forest(x_train, y_train, n_estimators=100):
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(x_train, y_train)
    return clf

def plot_variable_importance(model, clf, features, x_train, x_test, y_train, y_test):
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        if hasattr(clf, 'estimators_'):
            std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
        else:
            std = np.zeros_like(importances)
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_).flatten()
        std = np.zeros_like(importances)
    else:
        print(f"Using permutation importance for {type(clf).__name__}...")
        perm_importance = permutation_importance(clf, x_test, y_test, n_repeats=10, random_state=1)
        importances = perm_importance.importances_mean
        std = perm_importance.importances_std
    
    sorted_idx = np.argsort(importances)[::-1]
    padding = np.arange(x_train.size/len(x_train)) + 0.5
    plt.barh(padding, importances[sorted_idx], xerr=std[sorted_idx], align='center')
    plt.yticks(padding, features[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.show()

    scores = np.zeros(x_train.shape[1] + 1)
    for f in np.arange(0, x_train.shape[1] + 1) :
        x_train_reduced = x_train[:, sorted_idx[:f+1]]
        x_test_reduced = x_test [:, sorted_idx[:f+1]]
        model.fit(x_train_reduced, y_train)
        y_pred = model.predict(x_test_reduced)
        scores[f] = np.round(accuracy_score(y_test, y_pred), 3)
    
    optimal_features_count = np.argmax(scores) + 1
    print(f"Nombre optimal de variables : {optimal_features_count}")
    plt.plot(scores)
    plt.xlabel("Nombre de Variables")
    plt.ylabel("Accuracy")
    plt.title("Evolution de la moyenne de l'accuracy et la précision en fonction des variables")
    plt.show()
    return scores, optimal_features_count

def tune_classifier(param_grid, features, model, x_train, x_test, y_train, y_test, clf, verbose=False):
    def avg_score(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        return (acc + prec) / 2

    scorer = make_scorer(avg_score)

    x_train_scaled, x_test_scaled = standardise(x_train, x_test)
    _, optimal_features_count = plot_variable_importance(model, clf, features, x_train_scaled, x_test_scaled, y_train, y_test)

    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=verbose,
        n_jobs=-1
    )

    if not hasattr(clf, 'feature_importances_') and not hasattr(clf, 'coef_'):
        selector = SelectKBest(f_classif, k=optimal_features_count)
        x_train_reduced = selector.fit_transform(x_train, y_train)
    else:
        SFM = SelectFromModel(clf, max_features=optimal_features_count, prefit=True)
        SFM.fit(x_train, y_train)
        x_train_reduced = SFM.transform(x_train)

    grid_search.fit(x_train_reduced, y_train)

    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleur score (Accuracy & Precision) :", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    return grid_search.best_params_

def make_pipeline(filename, x_train, y_train, x_test=[], scale=True, pca=False, clf=MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1)):
    pipeline = Pipeline([
        ('scaler', StandardScaler() if scale else None),
        ('pca', PCA(n_components=3) if pca else None),
        ('classifier', clf)
    ])

    pipeline.fit(x_train, y_train)

    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)

    return pipeline.predict(x_test)

def load_pipeline(filename, x_test, y_test, verbose=False):
    with open(filename, 'rb') as f:
        loaded_pipeline = pickle.load(f)

    new_predictions = loaded_pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, new_predictions)
    precision = precision_score(y_test, new_predictions, zero_division=1)
    
    if verbose:
        print("Accuracy:", accuracy)
        print("Precision:", precision)
    return accuracy, precision

def run_classifiers(X, Y, clfs, n_splits=10, verbose=False):
    results = {}
    best_models = {'accuracy': None, 'precision': None, 'auc': None}
    best_scores = {'accuracy': -float('inf'), 'precision': -float('inf'), 'auc': -float('inf')}
    average_scores = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    for name, clf in clfs.items():
        start_time = time.time()
        
        cv_acc = cross_val_score(clf, X, Y, cv=kf, scoring='accuracy')
        mean_acc = np.mean(cv_acc)
        std_acc = np.std(cv_acc)
        
        try:
            cv_prec = cross_val_score(clf, X, Y, cv=kf, scoring='precision')
            mean_prec = np.mean(cv_prec)
            std_prec = np.std(cv_prec)
        except:
            mean_prec = None
            std_prec = None
        
        try:
            auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
            cv_auc = cross_val_score(clf, X, Y, cv=kf, scoring=auc_scorer)
            mean_auc = np.mean(cv_auc)
            std_auc = np.std(cv_auc)
        except:
            mean_auc = None
            std_auc = None
        
        exec_time = time.time() - start_time
        
        results[name] = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_precision': mean_prec,
            'std_precision': std_prec,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'execution_time': exec_time
        }

        if mean_acc > best_scores['accuracy']:
            best_scores['accuracy'] = mean_acc
            best_models['accuracy'] = clf

        if mean_prec is not None and mean_prec > best_scores['precision']:
            best_scores['precision'] = mean_prec
            best_models['precision'] = clf

        if mean_auc is not None and mean_auc > best_scores['auc']:
            best_scores['auc'] = mean_auc
            best_models['auc'] = clf

        metric_scores = [score for score in [mean_acc, mean_prec, mean_auc] if score is not None]
        avg_score = np.mean(metric_scores) if metric_scores else 0
        average_scores[name] = avg_score

        if verbose:
            print(f"{name}:")
            print(f"  Accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")
            if mean_prec is not None:
                print(f"  Precision: {mean_prec:.3f} +/- {std_prec:.3f}")
            if mean_auc is not None:
                print(f"  AUC: {mean_auc:.3f} +/- {std_auc:.3f}")
            print(f"  Execution Time: {exec_time:.2f} seconds\n")

    best_average_model_name = max(average_scores, key=average_scores.get)
    best_average_model_score = average_scores[best_average_model_name]
    best_average_model = clfs[best_average_model_name]

    for metric, model in best_models.items():
        print(f" Best model for {metric}: {model} [{best_scores[metric]:.3f}]")

    print(f"Best average model: ", best_average_model, best_average_model_score)

    return results, best_models, (best_average_model_score, best_average_model_name)
