import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# OutlierCapper Transformer
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            self.bounds[col] = (lower_bound, upper_bound)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            lower_bound, upper_bound = self.bounds.get(col, (None, None))
            if lower_bound is not None and upper_bound is not None:
                X_df[col] = np.where(X_df[col] < lower_bound, lower_bound, X_df[col])
                X_df[col] = np.where(X_df[col] > upper_bound, upper_bound, X_df[col])
        return X_df.values

# Load the trained model
import os
model_path = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')
model = joblib.load(model_path)


# Set page config
st.set_page_config(page_title="Heart Disease Predictor 💓", page_icon="💓", layout="wide")

# Background + Banner styling
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://static.vecteezy.com/system/resources/previews/043/993/968/non_2x/ai-generated-illustration-of-a-human-heart-in-art-style-for-medical-themes-photo.jpg");
        background-size: cover;
    }
    .banner {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 1rem 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# App Banner
st.markdown("""
    <div class="banner">
        <h1 style="color: #d90429;">💓 Heart Disease Risk Prediction 💓</h1>
        <p style="font-size: 1.1rem;">Enter your health details and instantly find out your risk level</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.pikbest.com/origin/09/17/59/48zpIkbEsTv7w.png!w700wp", use_column_width=True)
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This is an educational project built using:
- **Python**
- **Streamlit**
- **Machine Learning**

Model predicts your risk of heart disease based on your health inputs.  
Stay healthy! ❤️
""")

# Input Form
st.header("📋 Enter Patient Details:")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, value=120)
    cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=0, value=200)
    max_hr = st.number_input('Max Heart Rate Achieved', min_value=0, value=150)

with col2:
    sex = st.selectbox('Sex', ['M', 'F'])
    chest_pain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
    fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])

with col3:
    exercise_angina = st.selectbox('Exercise-induced Angina', ['Y', 'N'])
    oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, format="%.1f", value=1.0)
    st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

# Convert to DataFrame
input_data = {
    'Age': age,
    'Sex': sex,
    'ChestPainType': chest_pain,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'RestingECG': resting_ecg,
    'MaxHR': max_hr,
    'ExerciseAngina': exercise_angina,
    'Oldpeak': oldpeak,
    'ST_Slope': st_slope
}

input_df = pd.DataFrame([input_data])

# Predict button
if st.button('💓 Predict Risk'):
    prediction = model.predict(input_df)
    if prediction[0] == 0:
        st.success('🎉 No heart disease detected. Keep living healthy! ❤️')
    else:
        st.error('⚠️ Heart disease risk detected.This can lead to heart failure. Please consult your doctor immediately! 🏥')

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ by <strong>Ajinkya Itale</strong></center>", unsafe_allow_html=True)
