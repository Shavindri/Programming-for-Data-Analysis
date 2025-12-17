
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AQI Dashboard", layout="wide")

# ---------------- LOAD DATA ------------------
df = pd.read_csv("output/air_quality_dataset.csv")
model = joblib.load("models/aqi_model.pkl")

# ---------------- NAVIGATION -----------------
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Data Overview", "EDA", "AQI Prediction"]
)

# ================= HOME ======================
if page == "Home":
    st.title("Air Quality Analysis & AQI Prediction")
    st.write("This dashboard analyses air quality data and predicts AQI using a trained ML model.")

# ================= DATA OVERVIEW =============
elif page == "Data Overview":
    st.title("Dataset Overview")

    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Shape")
    st.write(df.shape)

    st.subheader("Columns")
    st.write(list(df.columns))

    st.subheader("Summary Statistics")
    numeric_df = df.select_dtypes(include="number")
    st.dataframe(numeric_df.describe())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

# ================= EDA =======================
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    numeric_cols = [
        'PM2.5','PM10','NO','NO2','NOx','NH3','CO',
        'SO2','O3','Benzene','Toluene','Xylene','AQI'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    st.subheader("AQI Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["AQI"].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("PM2.5 vs AQI")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df["PM2.5"], y=df["AQI"], ax=ax, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ================= PREDICTION =================
elif page == "AQI Prediction":
    st.title("AQI Prediction")

    features = [
        'PM2.5','PM10','NO','NO2','NOx','NH3','CO',
        'SO2','O3','Benzene','Toluene','Xylene'
    ]

    inputs = {}
    for f in features:
        inputs[f] = st.number_input(f, min_value=0.0, value=5.0)

    if st.button("Predict AQI"):
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted AQI: {prediction:.2f}")
