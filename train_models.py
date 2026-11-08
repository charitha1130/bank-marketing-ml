"""
Training script for the bank marketing classification assignment.
Trains 5 models on the same data, saves them in model/ and writes
a metrics table + test_data.csv for the streamlit app.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "bank-full.csv")
TARGET = "y"

# ---------- load data ----------
df = pd.read_csv(DATA_PATH, sep=";")
print("shape:", df.shape)
print(df[TARGET].value_counts())
print("missing values:", df.isnull().sum().sum())

# target as 0/1 (yes = client subscribed to term deposit)
df[TARGET] = (df[TARGET] == "yes").astype(int)

# ---------- train / test split ----------
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# keep the raw test rows (with target) so the app can take them as upload
os.makedirs("model", exist_ok=True)
test_data = X_test.copy()
test_data[TARGET] = y_test.values
test_data.to_csv("test_data.csv", index=False)

# ---------- preprocessing ----------
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
print("categorical:", cat_cols)
print("numeric:", num_cols)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# fit on training data only
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)
if hasattr(X_train_p, "toarray"):
    X_train_p = X_train_p.toarray()
    X_test_p = X_test_p.toarray()

joblib.dump(preprocessor, os.path.join("model", "preprocessor.pkl"))

# ---------- models ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                            random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE),
}

file_names = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}


def get_scores(model, X):
    # probability of class 1 for auc; svm has no predict_proba here
    # so use the decision function instead
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


results = []
for name, model in models.items():
    print("training", name, "...")
    model.fit(X_train_p, y_train)

    y_pred = model.predict(X_test_p)
    y_score = get_scores(model, X_test_p)

    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })

    joblib.dump(model, os.path.join("model", file_names[name]))

# ---------- results ----------
results_df = pd.DataFrame(results).round(4)
results_df.to_csv("model_results.csv", index=False)
print()
print(results_df.to_string(index=False))
