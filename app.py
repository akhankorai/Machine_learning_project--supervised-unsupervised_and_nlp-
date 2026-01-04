# app.py — Bank Marketing Subscription Predictor (Streamlit)
# ----------------------------------------------------------
# How to run:
#   1) Put this file next to the "artifacts/" folder created by the notebook
#   2) Install deps:  pip install -r requirements.txt
#      (minimum: streamlit, pandas, joblib, scikit-learn, imbalanced-learn)
#   3) Run:          streamlit run app.py
#
# Artifacts expected:
#   artifacts/bank_marketing_model.joblib
#   artifacts/bank_marketing_metadata.json  (optional but recommended)

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Bank Marketing — Term Deposit Predictor",
    page_icon="🏦",
    layout="wide",
)


# -------------------------
# Load model + metadata
# -------------------------
@st.cache_resource
def load_artifacts(artifacts_dir: str = "artifacts"):
    artifacts_path = Path(artifacts_dir)
    model_path = artifacts_path / "bank_marketing_model.joblib"
    meta_path = artifacts_path / "bank_marketing_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path.resolve()}\n"
            "Make sure you ran the notebook and it created artifacts/."
        )

    model = joblib.load(model_path)

    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception:
            metadata = {}

    # threshold fallback
    threshold = float(metadata.get("chosen_threshold", 0.5))

    return model, metadata, threshold


def add_features(X_in: pd.DataFrame) -> pd.DataFrame:
    """
    MUST match the notebook feature engineering exactly.
    Adds:
      - campaign_x_previous
      - was_previously_contacted
    """
    X = X_in.copy()

    if {"campaign", "previous"}.issubset(X.columns):
        # previous can be 0; +1 avoids zeroing out the interaction
        X["campaign_x_previous"] = X["campaign"] * (X["previous"] + 1)

    if "pdays" in X.columns:
        # in this dataset pdays == 999 means "not previously contacted"
        X["was_previously_contacted"] = (X["pdays"] != 999).astype(int)

    return X


def predict_one(model, row: dict, threshold: float):
    X = pd.DataFrame([row])
    X = add_features(X)

    # Most pipelines in the notebook used predict_proba (Calibrated for some models)
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= threshold)
    return pred, proba


# -------------------------
# UI helpers
# -------------------------
def _header():
    st.title("🏦 Bank Marketing — Term Deposit Subscription Predictor")
    st.caption(
        "Predict whether a customer will subscribe to a term deposit (`yes/no`) "
        "using the trained ML pipeline (preprocessing + SMOTE + best model)."
    )


def _model_info(metadata: dict, threshold: float):
    with st.expander("Model details (from metadata)", expanded=False):
        if not metadata:
            st.warning("No metadata file found (bank_marketing_metadata.json). Using default threshold = 0.5.")
            st.write({"threshold": threshold})
            return
        st.write(
            {
                "best_model": metadata.get("best_model", "unknown"),
                "chosen_threshold": threshold,
                "feature_engineering": metadata.get("feature_engineering", []),
            }
        )
        if "test_metrics" in metadata:
            st.subheader("Test metrics (saved from notebook)")
            st.json(metadata["test_metrics"])


def _single_prediction_form(model, threshold: float):
    st.subheader("Single prediction")

    # NOTE: These fields match the common "bank-additional-full" dataset.
    # If your training CSV differs, adjust field names accordingly.
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("age", min_value=15, max_value=100, value=35, step=1)
        job = st.selectbox(
            "job",
            [
                "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                "retired", "self-employed", "services", "student", "technician",
                "unemployed", "unknown",
            ],
            index=0,
        )
        marital = st.selectbox("marital", ["married", "single", "divorced", "unknown"], index=0)
        education = st.selectbox(
            "education",
            ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
             "professional.course", "university.degree", "unknown"],
            index=3,
        )

        default = st.selectbox("default", ["no", "yes", "unknown"], index=0)
        housing = st.selectbox("housing", ["no", "yes", "unknown"], index=1)
        loan = st.selectbox("loan", ["no", "yes", "unknown"], index=0)

    with col2:
        contact = st.selectbox("contact", ["cellular", "telephone"], index=0)
        month = st.selectbox("month", ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], index=2)
        day_of_week = st.selectbox("day_of_week", ["mon", "tue", "wed", "thu", "fri"], index=0)

        duration = st.number_input("duration (seconds)", min_value=0, max_value=6000, value=180, step=10)
        campaign = st.number_input("campaign", min_value=1, max_value=100, value=2, step=1)
        pdays = st.number_input("pdays (999 = not previously contacted)", min_value=0, max_value=999, value=999, step=1)
        previous = st.number_input("previous", min_value=0, max_value=50, value=0, step=1)

        poutcome = st.selectbox("poutcome", ["failure", "nonexistent", "success"], index=1)

    with col3:
        emp_var_rate = st.number_input("emp.var.rate", value=1.1, step=0.1, format="%.2f")
        cons_price_idx = st.number_input("cons.price.idx", value=93.99, step=0.01, format="%.2f")
        cons_conf_idx = st.number_input("cons.conf.idx", value=-36.4, step=0.1, format="%.2f")
        euribor3m = st.number_input("euribor3m", value=4.86, step=0.01, format="%.2f")
        nr_employed = st.number_input("nr.employed", value=5191.0, step=1.0, format="%.1f")

    row = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
    }

    left, right = st.columns([1, 2])
    with left:
        custom_threshold = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=float(threshold), step=0.01)

    if st.button("Predict", type="primary"):
        try:
            pred, proba = predict_one(model, row, custom_threshold)
        except Exception as e:
            st.error("Prediction failed. Check that your input columns match the training pipeline.")
            st.exception(e)
            return

        label = "YES (will subscribe)" if pred == 1 else "NO (will not subscribe)"
        st.metric("Prediction", label)
        st.metric("Probability (yes)", f"{proba:.4f}")

        st.info(
            "Tip: For imbalanced problems, a threshold ≠ 0.50 can improve business outcomes "
            "(e.g., maximize F1, or target higher precision/recall)."
        )

        with st.expander("Show input row used for prediction"):
            st.dataframe(pd.DataFrame([row]))


def _batch_predictor(model, threshold: float):
    st.subheader("Batch prediction (CSV upload)")

    st.write(
        "Upload a CSV containing the **same feature columns** used in training (excluding `y`). "
        "The app will output predictions + probabilities."
    )

    up = st.file_uploader("Upload CSV", type=["csv"])
    if not up:
        return

    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error("Could not read the CSV.")
        st.exception(e)
        return

    st.write("Preview:")
    st.dataframe(df.head(10))

    thr = st.slider("Batch threshold", min_value=0.05, max_value=0.95, value=float(threshold), step=0.01, key="batch_thr")

    if st.button("Run batch prediction"):
        try:
            X = add_features(df)
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= thr).astype(int)

            out = df.copy()
            out["pred_label"] = np.where(preds == 1, "yes", "no")
            out["prob_yes"] = probs

            st.success("Done!")
            st.dataframe(out.head(50))

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="bank_marketing_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error("Batch prediction failed. Check column names/dtypes.")
            st.exception(e)


# -------------------------
# Main
# -------------------------
def main():
    _header()

    try:
        model, metadata, threshold = load_artifacts("artifacts")
    except Exception as e:
        st.error("Could not load model artifacts.")
        st.exception(e)
        st.stop()

    _model_info(metadata, threshold)

    tab1, tab2 = st.tabs(["🧍 Single prediction", "📄 Batch prediction (CSV)"])
    with tab1:
        _single_prediction_form(model, threshold)
    with tab2:
        _batch_predictor(model, threshold)


if __name__ == "__main__":
    main()
