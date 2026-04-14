import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time

# ==============================
# CONFIGURATION DE LA PAGE
# ==============================
st.set_page_config(
    page_title="CardioDiag - IRM Cardiaque",
    page_icon="🫀",
    layout="wide"
)

# ==============================
# PRÉTRAITEMENT
# ==============================
def custom_preprocess(img_array):
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype("uint8")
    else:
        img_array = img_array.astype("uint8")

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img.astype("float32") / 255.0
    return img

# ==============================
# CHARGEMENT DU MODÈLE
# ==============================
@st.cache_resource
def load_model():
    model_path = "model.keras"
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            # إضافة compile=False تسرع التحميل وتتجنب أخطاء التوقيع
            model = tf.keras.models.load_model(model_path, compile=False)
            return model, True
        except Exception as e:
            st.error(f"⚠️ Erreur : {e}")
            return None, False

model, is_loaded = load_model()

# ==============================
# INTERFACE PRINCIPALE
# ==============================
st.title("🫀 CardioDiag - Diagnostic IA")

with st.sidebar:
    st.header("⚙️ Paramètres")
    threshold = st.slider("Seuil de décision", 0.0, 1.0, 0.5, 0.01)
    if is_loaded:
        st.success("✅ Modèle réel chargé")
    else:
        st.error("❌ Modèle non chargé")

uploaded_file = st.file_uploader("📂 Charger une image IRM", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and is_loaded:
    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)
    
    with st.spinner("Analyse en cours..."):
        processed = custom_preprocess(original)
        # التنفيذ الحقيقي للتنبؤ
        input_tensor = np.expand_dims(processed, axis=0)
        proba = float(model.predict(input_tensor, verbose=0)[0][0])
        
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Image Originale", use_container_width=True)
    with col2:
        st.image(processed, caption="Image Prétraitée", use_container_width=True)

    # التحليل والنتائج
    st.markdown("---")
    diagnosis = "CAD (Malade)" if proba >= threshold else "Normal (Sain)"
    
    col_res1, col_res2 = st.columns(2)
    col_res1.metric("Diagnostic Final", diagnosis)
    col_res2.metric("Probabilité CAD", f"{proba:.2%}")

    if diagnosis == "CAD (Malade)":
        st.warning(f"⚠️ Alerte : Probabilité de maladie détectée ({proba:.2%})")
    else:
        st.success(f"✅ Résultat : Caractéristiques normales ({1-proba:.2%})")
