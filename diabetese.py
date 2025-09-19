import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Diabetes Predictor", page_icon="💉")

st.title("💉 Diabetes Prediction App")
st.markdown(
    "Upload a **CSV dataset** (like the Kaggle *Pima Indians Diabetes Database*). "
    "It must contain all the standard columns including `Outcome`."
)

# 1️⃣ Upload the dataset
uploaded_file = st.file_uploader("Upload diabetes.csv", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # 2️⃣ Train the model
    if "Outcome" not in df.columns:
        st.error("The uploaded file must have an 'Outcome' column (0 = No diabetes, 1 = Diabetes).")
    else:
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=500)
        model.fit(X_scaled, y)

        st.success("Model trained successfully! Enter patient details below:")

        # 3️⃣ User input for prediction
        preg = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 200, 120)
        bp = st.number_input("BloodPressure", 0, 150, 70)
        skin = st.number_input("SkinThickness", 0, 99, 20)
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.number_input("Age", 1, 120, 33)

        if st.button("Predict"):
            user_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
            user_data = scaler.transform(user_data)
            pred = model.predict(user_data)
            prob = model.predict_proba(user_data)[0][1]
            st.success("Result: **Diabetic**" if pred[0] == 1 else "Result: **Not Diabetic**")
            st.info(f"Model confidence: {prob*100:.1f}%")
else:
    st.info("👆 Please upload the dataset to start.")
