import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ================= TENSORFLOW SAFETY =================
tf.keras.backend.clear_session()

# ================= LOAD MODEL SAFELY =================
@st.cache_resource
def load_churn_model():
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model("model.h5", compile=False)
    return model

model = load_churn_model()

# ================= LOAD ENCODERS =================
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ================= STREAMLIT UI =================
st.title("Customer Churn Prediction")

# ---- Inputs ----
geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0, step=100.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0)
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# ================= PREDICTION =================
if st.button("Predict Churn"):

    # ---- Prepare input ----
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    final_input = pd.concat(
        [input_data.reset_index(drop=True),
         geo_encoded_df.reset_index(drop=True)],
        axis=1
    )

    # ---- Scale ----
    input_scaled = scaler.transform(final_input).astype(np.float32)

    # ---- Predict (SAFE WAY) ----
    prediction = model(input_scaled, training=False).numpy()

    if prediction.ndim == 2:
        churn_probability = float(prediction[0][0])
    else:
        churn_probability = float(prediction[0])

    # ---- Output ----
    st.subheader("Prediction Result")
    st.write(f"Churn Probability: **{churn_probability:.2f}**")

    if churn_probability > 0.5:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is not likely to churn.")