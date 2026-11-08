import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Bank Term Deposit Predictor", layout="wide")

TARGET = "y"
MODEL_DIR = "model"

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}


@st.cache_resource
def load_model(file_name):
    return joblib.load(os.path.join(MODEL_DIR, file_name))


@st.cache_resource
def load_preprocessor():
    return joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


# ---------------- header ----------------
st.title("Bank Marketing - Term Deposit Prediction")
st.write(
    "Pick one of the five trained models, upload the test csv and see how "
    "it does. The models are already trained, nothing is retrained here."
)

# ---------------- sidebar ----------------
st.sidebar.header("Options")
model_name = st.sidebar.selectbox("Select a model", list(model_files.keys()))
uploaded_file = st.sidebar.file_uploader("Upload test data (csv)", type=["csv"])
st.sidebar.caption(
    "The file needs all 16 input columns plus the target column 'y' "
    "(0/1 or no/yes). Use test_data.csv from the repo."
)

if uploaded_file is None:
    st.info("Upload a csv from the sidebar to get started.")
    st.stop()

# ---------------- read data ----------------
try:
    data = pd.read_csv(uploaded_file)
    if data.shape[1] == 1:
        # the original UCI file uses ';' as separator
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file, sep=";")
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

preprocessor = load_preprocessor()
needed_cols = [c for name, _, cols in preprocessor.transformers_
               if name != "remainder" for c in cols]

missing = [c for c in needed_cols + [TARGET] if c not in data.columns]
if missing:
    st.error("These columns are missing in the uploaded file: " + ", ".join(missing))
    st.stop()

y_true = data[TARGET]
if y_true.dtype == object:
    y_true = (y_true.str.strip().str.lower() == "yes").astype(int)
y_true = y_true.astype(int)

X = data[needed_cols]

st.subheader("Uploaded data")
st.write(f"{data.shape[0]} rows, {data.shape[1]} columns")
st.dataframe(data.head(10), use_container_width=True)

# ---------------- predictions ----------------
model = load_model(model_files[model_name])
X_processed = preprocessor.transform(X)
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

y_pred = model.predict(X_processed)
y_score = get_scores(model, X_processed)

# ---------------- metrics ----------------
st.subheader(f"Evaluation metrics - {model_name}")

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_score) if y_true.nunique() > 1 else float("nan"),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall": recall_score(y_true, y_pred, zero_division=0),
    "F1": f1_score(y_true, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

cols = st.columns(len(metrics))
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

# ---------------- confusion matrix / report ----------------
left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["no", "yes"], yticklabels=["no", "yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(y_true, y_pred, target_names=["no", "yes"],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ---------------- all models on the same data ----------------
st.subheader("All models on this data")
rows = []
for name, f in model_files.items():
    m = load_model(f)
    p = m.predict(X_processed)
    s = get_scores(m, X_processed)
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_true, p),
        "AUC": roc_auc_score(y_true, s) if y_true.nunique() > 1 else float("nan"),
        "Precision": precision_score(y_true, p, zero_division=0),
        "Recall": recall_score(y_true, p, zero_division=0),
        "F1": f1_score(y_true, p, zero_division=0),
        "MCC": matthews_corrcoef(y_true, p),
    })
st.dataframe(pd.DataFrame(rows).round(4).set_index("ML Model Name"),
             use_container_width=True)
