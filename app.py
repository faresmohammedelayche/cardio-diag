# ==============================
# app.py - Optimized Version
# CardioDiag - IRM Cardiaque
# ==============================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import random

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 224

st.set_page_config(
    page_title="CardioDiag - IRM Cardiaque",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CSS
# ==============================
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    text-align: center;
    margin-bottom: 2rem;
    color: gray;
}
.footer {
    text-align: center;
    margin-top: 3rem;
    font-size: 0.8rem;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# IMAGE PREPROCESSING
# ==============================
def resize_image(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

def custom_preprocess(img_array):
    img = resize_image(img_array)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img.astype(np.float32) / 255.0

@st.cache_data
def process_image_cached(img_array):
    return custom_preprocess(img_array)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    model_path = "model.keras"

    if os.path.exists(model_path):
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(model_path)
            model.compile(run_eagerly=False)

            return model, True
        except Exception as e:
            st.warning(f"⚠️ Model loading error: {e}")
            return None, False
    else:
        return None, False

model, is_real = load_model()

# ==============================
# PREDICTION
# ==============================
def predict(processed_img):
    if not is_real:
        random.seed(42)
        return random.uniform(0.2, 0.8)

    import numpy as np

    input_tensor = np.expand_dims(processed_img, axis=0)
    preds = model(input_tensor, training=False).numpy()

    # ===== AUTO DETECT OUTPUT =====
    if preds.shape[-1] == 1:
        # sigmoid
        proba = float(preds[0][0])

    elif preds.shape[-1] == 2:
        # softmax
        proba = float(preds[0][1])  # CAD class

    else:
        raise ValueError(f"Unexpected output shape: {preds.shape}")

    return proba

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.title("⚙️ Paramètres")

    threshold = st.slider(
        "Seuil de décision",
        0.0, 1.0, 0.5, 0.01
    )

    st.markdown("---")

    st.subheader("Validation (optionnelle)")
    true_label = st.radio(
        "Classe réelle",
        ["Non renseignée", "Normal", "CAD"]
    )

    st.markdown("---")

    if is_real:
        st.success("Modèle réel chargé")
    else:
        st.warning("Mode simulation")

# ==============================
# MAIN UI
# ==============================
st.markdown('<div class="main-header">🫀 CardioDiag</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Diagnostic IRM Cardiaque assisté par IA</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Charger une image IRM",
        type=["png", "jpg", "jpeg", "bmp", "tiff"]
    )

# ==============================
# PROCESS
# ==============================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)

    processed = process_image_cached(original)

    # Display images
    with col2:
        st.markdown("### Visualisation")
        c1, c2 = st.columns(2)
        with c1:
            st.image(original, caption="Original", use_container_width=True)
        with c2:
            st.image(processed, caption="Prétraité", use_container_width=True)

    # Prediction
    proba = predict(processed)

    if proba >= threshold:
        diagnosis = "CAD (maladie coronarienne)"
        confidence = proba
    else:
        diagnosis = "Normal"
        confidence = 1 - proba

    # Results
    st.markdown("---")
    st.markdown("## Résultat")

    m1, m2, m3 = st.columns(3)

    m1.metric("Diagnostic", diagnosis)
    m2.metric("Probabilité CAD", f"{proba:.2%}")
    m3.metric("Confiance", f"{confidence:.2%}")

    st.progress(proba)

    if diagnosis.startswith("CAD"):
        st.warning("Consultation médicale recommandée")
    else:
        st.success("Image normale")

    # ==========================
    # VALIDATION
    # ==========================
    if true_label != "Non renseignée":
        st.markdown("---")

        expected = true_label
        predicted = "CAD" if "CAD" in diagnosis else "Normal"

        if expected == predicted:
            st.success("Prédiction correcte")
        else:
            st.error("Prédiction incorrecte")

else:
    with col2:
        st.info("Chargez une image pour commencer")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("""
<div class="footer">
Application éducative - ne remplace pas un médecin
</div>
""", unsafe_allow_html=True)
