import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Streamlit setup ---
st.set_page_config(page_title="Post-COVID Risk Predictor", layout="centered")
st.title("🧠 Post-COVID (Long-COVID) Risk Prediction")

st.write("A simple demo app that predicts the likelihood of long-COVID using basic health data.")
st.write("⚠️ Educational use only — not for medical diagnosis.")

# --- Generate synthetic data ---
def generate_data(n=1000, seed=42):
    np.random.seed(seed)
    age = np.random.randint(18, 80, n)
    bmi = np.random.normal(26, 5, n)
    diabetes = np.random.binomial(1, 0.1, n)
    hypertension = np.random.binomial(1, 0.2, n)
    severe_case = np.random.binomial(1, 0.15, n)
    vaccinated = np.random.choice(["none", "partial", "full", "booster"], n, p=[0.2,0.2,0.4,0.2])

    # Synthetic rule: higher age + comorbidities = more risk
    risk_score = 0.03*(age-40) + 0.8*diabetes + 0.5*hypertension + 1.0*severe_case - 0.7*(vaccinated=="booster")
    prob = 1 / (1 + np.exp(-risk_score))
    long_covid = np.random.binomial(1, prob)

    df = pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "diabetes": diabetes,
        "hypertension": hypertension,
        "severe_case": severe_case,
        "vaccinated": vaccinated,
        "long_covid": long_covid
    })
    return df

# --- Train model ---
df = generate_data()
X = pd.get_dummies(df.drop("long_covid", axis=1), drop_first=True)
y = df["long_covid"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --- Symptom prediction logic ---
def predict_symptoms(risk_percent):
    if risk_percent <= 20:
        return ["Occasional fatigue"]
    elif risk_percent <= 40:
        return ["Mild fatigue", "Occasional breath problem", "Minor joint pain"]
    elif risk_percent <= 60:
        return ["Moderate fatigue", "Mild chest discomfort", "Mild headache", "Mild breath issues"]
    elif risk_percent <= 80:
        return ["Strong fatigue", "Chest pain", "Breath difficulty", "Body aches", "Dizziness"]
    else:
        return ["Severe fatigue", "Severe chest pain", "Breath difficulty", "Heart strain", "Joint pains", "Dizziness"]

# --- Input form ---
st.subheader("Enter Patient Details")

age = st.slider("Age", 18, 90, 40)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
diabetes = st.selectbox("Diabetes", [0, 1])
hypertension = st.selectbox("Hypertension", [0, 1])
severe_case = st.selectbox("Initial COVID severity (Severe case)", [0, 1])
vaccinated = st.selectbox("Vaccination status", ["none", "partial", "full", "booster"])

# --- Prediction ---
if st.button("Predict Risk"):
    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "diabetes": diabetes,
        "hypertension": hypertension,
        "severe_case": severe_case,
        "vaccinated": vaccinated
    }])
    input_encoded = pd.get_dummies(input_data, drop_first=True).reindex(columns=X.columns, fill_value=0)
    prob = model.predict_proba(input_encoded)[0, 1]
    risk_percent = prob * 100

    st.success(f"Predicted Long-COVID Risk: **{risk_percent:.1f}%**")

    # 👇 Call your symptom prediction function here
    likely_pains = predict_symptoms(risk_percent)
    st.subheader("Likely Symptoms (Predicted):")
    for s in likely_pains:
        st.write(f"• {s}")

st.caption("🧩 This is a simple AI demo — for learning and experimentation only.")
