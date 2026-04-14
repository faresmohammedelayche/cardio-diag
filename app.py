# ==============================
# CardioDiag - FINAL VERSION
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
st.set_page_config(
    page_title="CardioDiag",
    page_icon="🫀",
    layout="wide"
)

# ==============================
# PREPROCESSING (MATCH TRAINING)
# ==============================
def custom_preprocess(img_array):
    # Resize to 512x512
    img = cv2.resize(img_array, (512, 512), interpolation=cv2.INTER_LINEAR)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Normalize
    normalized = enhanced.astype(np.float32) / 255.0

    # Expand dims → (512,512,1)
    normalized = np.expand_dims(normalized, axis=-1)

    return normalized

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
            return model, True
        except Exception as e:
            st.error(f"Model error: {e}")
            return None, False
    return None, False

model, is_real = load_model()

# ==============================
# PREDICTION
# ==============================
def predict(processed_img):
    if not is_real:
        return random.uniform(0.2, 0.8)

    input_tensor = np.expand_dims(processed_img, axis=0)
    preds = model(input_tensor, training=False).numpy()

    if preds.shape[-1] == 1:
        return float(preds[0][0])
    elif preds.shape[-1] == 2:
        return float(preds[0][1])
    else:
        raise ValueError(f"Unexpected output shape: {preds.shape}")

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.title("⚙️ Paramètres")

    threshold = st.slider("Seuil de décision", 0.0, 1.0, 0.5, 0.01)

    st.markdown("---")

    if is_real:
        st.success("✅ Modèle chargé")
    else:
        st.warning("⚠️ Mode simulation")

# ==============================
# MAIN UI
# ==============================
st.title("🫀 CardioDiag")
st.caption("Diagnostic IRM Cardiaque par IA")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "📂 Charger une image IRM",
        type=["png", "jpg", "jpeg", "bmp", "tiff"]
    )

# ==============================
# PROCESS
# ==============================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)

    processed = custom_preprocess(original)

    # DISPLAY
    with col2:
        st.subheader("Visualisation")
        c1, c2 = st.columns(2)

        with c1:
            st.image(original, caption="Original", use_container_width=True)

        with c2:
            st.image(processed.squeeze(), caption="Prétraité", use_container_width=True)

    # PREDICTION
    proba = predict(processed)

    # DECISION LOGIC (CORRECT)
    is_cad = proba >= threshold

    diagnosis = "CAD (maladie coronarienne)" if is_cad else "Normal"
    confidence = proba if is_cad else (1 - proba)

    # RESULTS
    st.markdown("---")
    st.subheader("Résultat")

    c1, c2, c3 = st.columns(3)
    c1.metric("Diagnostic", diagnosis)
    c2.metric("Probabilité CAD", f"{proba:.2%}")
    c3.metric("Confiance", f"{confidence:.2%}")

    st.progress(proba)

    if is_cad:
        st.warning("⚠️ Consultation médicale recommandée")
    else:
        st.success("✅ Image normale")

    # DEBUG (يمكن حذفه لاحقًا)
    st.markdown("---")
    st.caption("Debug info")
    st.write("Probability:", proba)
    st.write("Threshold:", threshold)
    st.write("Processed mean:", processed.mean())

else:
    with col2:
        st.info("👈 قم برفع صورة للبدء")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("⚠️ هذا التطبيق لأغراض تعليمية فقط")
