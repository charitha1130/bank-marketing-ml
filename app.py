import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Term Deposit Predictor", page_icon="🏦",
                   layout="wide")

TARGET = "y"
MODEL_DIR = "model"
SAMPLE_FILE = "test_data.csv"

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}

# a bit of styling for the cards
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    .hero {
        background: linear-gradient(120deg, #0E7490 0%, #4F46E5 60%, #9333EA 100%);
        padding: 26px 32px;
        border-radius: 14px;
        margin-bottom: 1.2rem;
    }
    .hero h1 {color: white; margin: 0; font-size: 2rem; padding: 0;}
    .hero p {color: #E0E7FF; margin: 6px 0 0 0; font-size: 1rem;}
    div[data-testid="stMetric"] {
        background-color: white;
        border-top: 4px solid #4F46E5;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.12);
    }
    div[data-testid="stMetricLabel"] p {font-size: 0.9rem; color: #64748B;}
    div[data-testid="stMetricValue"] {color: #1E293B;}
    button[data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
    section[data-testid="stSidebar"] {border-right: 3px solid #C7D2FE;}
    </style>
    """,
    unsafe_allow_html=True,
)


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


def get_metrics(y_true, y_pred, y_score):
    both_classes = len(np.unique(y_true)) > 1
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score) if both_classes else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


@st.cache_data(show_spinner="Running the saved models...")
def run_all_models(X_processed, y_true):
    out = {}
    for name, f in model_files.items():
        m = load_model(f)
        pred = m.predict(X_processed)
        score = get_scores(m, X_processed)
        out[name] = {"pred": pred, "score": score,
                     "metrics": get_metrics(y_true, pred, score)}
    return out


# ---------------- sidebar ----------------
st.sidebar.title("Controls")
model_name = st.sidebar.selectbox("Model", list(model_files.keys()))
uploaded_file = st.sidebar.file_uploader("Test data (csv)", type=["csv"])
st.sidebar.caption(
    "Needs the 16 input columns and the target column `y` (0/1 or no/yes). "
    "The models are loaded from disk and are not retrained."
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: UCI Bank Marketing (Moro et al., 2014)")

# ---------------- header ----------------
st.markdown(
    """
    <div class="hero">
      <h1>🏦 Bank Marketing - Term Deposit Prediction</h1>
      <p>Five classifiers trained on a Portuguese bank's phone campaign data.
      Choose a model on the left, upload the test file and check how it scores.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- read data ----------------
if uploaded_file is not None:
    source = uploaded_file
    try:
        data = pd.read_csv(source)
        if data.shape[1] == 1:
            # original UCI file is ';' separated
            source.seek(0)
            data = pd.read_csv(source, sep=";")
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()
elif os.path.exists(SAMPLE_FILE):
    data = pd.read_csv(SAMPLE_FILE)
    st.info("No file uploaded yet, so the bundled test_data.csv is shown. "
            "Upload a csv from the sidebar to use your own.")
else:
    st.info("Upload a csv from the sidebar to get started.")
    st.stop()

preprocessor = load_preprocessor()
needed_cols = [c for name, _, cols in preprocessor.transformers_
               if name != "remainder" for c in cols]

missing = [c for c in needed_cols + [TARGET] if c not in data.columns]
if missing:
    st.error("These columns are missing in the file: " + ", ".join(missing))
    st.stop()

y_true = data[TARGET]
if y_true.dtype == object:
    y_true = (y_true.str.strip().str.lower() == "yes").astype(int)
y_true = y_true.astype(int).to_numpy()

X_processed = preprocessor.transform(data[needed_cols])
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

all_results = run_all_models(X_processed, y_true)
chosen = all_results[model_name]

# ---------------- summary cards ----------------
c1, c2, c3 = st.columns(3)
c1.metric("Rows in file", f"{len(data):,}")
c2.metric("Clients who subscribed", f"{y_true.mean() * 100:.1f}%")
c3.metric("Input features", len(needed_cols))

tab_result, tab_compare, tab_data = st.tabs(
    ["📊 Selected model", "⚖️ Compare all models", "🗂️ Data preview"])

# ---------------- tab 1: selected model ----------------
with tab_result:
    st.subheader(model_name)

    cols = st.columns(6)
    for col, (k, v) in zip(cols, chosen["metrics"].items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("##### Confusion matrix")
        cm = confusion_matrix(y_true, chosen["pred"])
        row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        labels = np.array([[f"{cm[i, j]}\n({row_pct[i, j]:.1f}%)"
                            for j in range(2)] for i in range(2)])
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        sns.heatmap(cm, annot=labels, fmt="", cmap="PuBu", cbar=False, ax=ax,
                    linewidths=2, linecolor="white",
                    annot_kws={"fontsize": 12, "fontweight": "bold"},
                    xticklabels=["no", "yes"], yticklabels=["no", "yes"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, use_container_width=False)
        caught = cm[1, 1]
        st.caption(
            f"Of {cm[1].sum()} clients who really subscribed, the model "
            f"found {caught} and missed {cm[1, 0]}. "
            f"It also flagged {cm[0, 1]} clients who did not subscribe.")

    with right:
        st.markdown("##### Classification report")
        report = classification_report(y_true, chosen["pred"],
                                       target_names=["no", "yes"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ---------------- tab 2: compare ----------------
with tab_compare:
    st.subheader("All five models on this file")
    comp = pd.DataFrame(
        {name: res["metrics"] for name, res in all_results.items()}).T
    comp.index.name = "ML Model Name"
    st.dataframe(
        comp.style.format("{:.4f}").highlight_max(axis=0, color="#BAE6FD"),
        use_container_width=True)
    st.caption("Best value in each column is highlighted.")

    pick = st.multiselect("Metrics to plot", list(comp.columns),
                          default=["Accuracy", "AUC", "F1", "MCC"])
    if pick:
        st.bar_chart(comp[pick])

# ---------------- tab 3: data ----------------
with tab_data:
    st.subheader("Uploaded data")
    st.write(f"{data.shape[0]} rows, {data.shape[1]} columns")
    st.dataframe(data.head(50), use_container_width=True)
