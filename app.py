"""
Parkinson's Disease Detection - Streamlit Web App
====================================================
Yeh web app pehle se train kiye hue model ko load karta hai
aur user se voice-features input le kar Parkinson's disease
predict karta hai.

IMPORTANT: Isse chalane se PEHLE 'parkinsons_detection.py' ek
baar chala lena, taaki model save ho jaye (.pkl files banein).

Chalane ka tareeka:
    streamlit run app.py

Requirements:
    pip install streamlit joblib numpy pandas
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import joblib
import os

# -----------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Parkinson's Disease Detector",
    page_icon="🧠",
    layout="centered"
)

MODEL_FILE = "parkinsons_model.pkl"
SCALER_FILE = "parkinsons_scaler.pkl"
COLUMNS_FILE = "parkinsons_columns.pkl"


# -----------------------------------------------------------------
# Model, Scaler aur Columns load karo (cache karke, taaki baar baar
# load na ho aur app fast chale)
# -----------------------------------------------------------------
@st.cache_resource
def load_model_files():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(COLUMNS_FILE)):
        return None, None, None
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    columns = joblib.load(COLUMNS_FILE)
    return model, scaler, columns


model, scaler, feature_columns = load_model_files()


# -----------------------------------------------------------------
# Title aur Description
# -----------------------------------------------------------------
st.title("🧠 Parkinson's Disease Detector")
st.write(
    "Yeh app voice-recording se nikale gaye features ke basis par "
    "Parkinson's disease detect karta hai, XGBoost machine learning model use karke."
)

# -----------------------------------------------------------------
# Real photo dikhane ke liye (Pexels/Pixabay/Unsplash se free photo
# download karke "disease_photo.jpg" naam se isi folder mein rakho)
# -----------------------------------------------------------------
PHOTO_FILE = "disease_photo.jpg"

if os.path.exists(PHOTO_FILE):
    st.image(PHOTO_FILE, use_container_width=True, caption="Parkinson's Disease Awareness")
else:
    st.info(
        "pexels-matthew-hamilton-1620041.jpg photo file nahi mili. "
        f"se download karke is folder mein '{PHOTO_FILE}' naam se save kar do."
    )

# -----------------------------------------------------------------
# Parkinson's Disease ke common symptoms
# -----------------------------------------------------------------
st.subheader("🩺 Parkinson's Disease ke Common Symptoms")

sym_col1, sym_col2 = st.columns(2)

with sym_col1:
    st.markdown(
        """
        - 🤲 **Tremors (kaanpna)** — haath, ungliyon ya thodi mein halka kaanpna, aksar rest karte waqt
        - 🦴 **Muscle Stiffness** — shareer ke jodo/muscles mein akadan, movement mein dikkat
        - 🐢 **Bradykinesia** — movements ka dheere-dheere hona, chalne mein time lagna
        - ⚖️ **Balance Problems** — santulan banaye rakhne mein dikkat, girne ka risk
        """
    )

with sym_col2:
    st.markdown(
        """
        - 🗣️ **Speech Changes** — awaaz dhimi, kaanpti ya monotone ho jaana
        - ✍️ **Micrographia** — likhawat chhoti aur close ho jaana
        - 😐 **Reduced Facial Expressions** — chehre ke bhaav kam ho jaana
        - 😴 **Sleep Issues** — neend mein dikkat, chalte-chalte awaaz badalna
        """
    )

st.info(
    "ℹ️ **Note:** Yeh symptoms dheere-dheere, kai saalon mein develop hote hain. "
    "Agar aapko ya kisi apne ko yeh lakshan lagatar dikh rahe hain, "
    "turant kisi neurologist se consult karo — early diagnosis se treatment behtar hota hai."
)

# -----------------------------------------------------------------
# Agar model files nahi mili, toh warning dikhao
# -----------------------------------------------------------------
if model is None:
    st.error(
        "⚠️ Model files nahi mili! Pehle terminal mein yeh command chalao:\n\n"
        "`python parkinsons_detection.py`\n\n"
        "Isse model train hoga aur save ho jayega. Uske baad is app ko refresh karo."
    )
    st.stop()

st.success("✅ Model successfully load ho gaya hai. Neeche values daal ke predict karo.")

st.divider()

# -----------------------------------------------------------------
# Do tareeke diye hain input dene ke:
# 1. Manually har feature ki value daalna
# 2. Sample/random values bhar dena (testing ke liye)
# -----------------------------------------------------------------
st.subheader("📊 Voice Features Enter Karo")

with st.expander("ℹ️ Yeh features kya hain?"):
    st.write(
        "Yeh saare features UCI Parkinsons dataset se hain, jo ek insaan ki "
        "voice recording se nikale gaye hote hain (jaise pitch, frequency variation, "
        "amplitude variation, etc). Agar aapke paas exact values nahi hain, toh "
        "'Load Sample Values' button use karo test karne ke liye."
    )

# Sample values (dataset ka ek healthy aur ek disease wala record - testing ke liye)
sample_healthy = [
    197.076, 206.896, 192.055, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498,
    0.01098, 0.097, 0.00563, 0.0068, 0.00802, 0.01689, 0.00339, 26.775,
    0.422229, 0.741367, -7.3483, 0.177551, 1.743867, 0.085569
]

sample_disease = [
    119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109,
    0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033,
    0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654
]

col1, col2 = st.columns(2)
with col1:
    if st.button("🟢 Load Healthy Sample"):
        st.session_state["values"] = sample_healthy
with col2:
    if st.button("🔴 Load Disease Sample"):
        st.session_state["values"] = sample_disease

if "values" not in st.session_state:
    st.session_state["values"] = [0.0] * len(feature_columns)

# -----------------------------------------------------------------
# Har feature ke liye number input banayein
# -----------------------------------------------------------------
input_values = []
cols = st.columns(2)
for idx, col_name in enumerate(feature_columns):
    with cols[idx % 2]:
        val = st.number_input(
            col_name,
            value=float(st.session_state["values"][idx]),
            format="%.6f",
            key=f"input_{idx}"
        )
        input_values.append(val)

st.divider()

# -----------------------------------------------------------------
# Predict Button
# -----------------------------------------------------------------
if st.button("🔍 Predict Karo", type="primary", use_container_width=True):
    # Input ko numpy array mein badlo aur scale karo
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Prediction lo
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    st.divider()
    st.subheader("📋 Result")

    if prediction == 1:
        st.error(
            f"⚠️ **Parkinson's Disease Detected**\n\n"
            f"Confidence: {prediction_proba[1] * 100:.2f}%"
        )
    else:
        st.success(
            f"✅ **Healthy (No Parkinson's Disease Detected)**\n\n"
            f"Confidence: {prediction_proba[0] * 100:.2f}%"
        )

    st.caption(
        "⚕️ Disclaimer: Yeh sirf ek educational ML project hai. "
        "Kisi bhi real medical decision ke liye asli doctor se consult karo."
    )