import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# يجب أن يكون هذا أول أمر في الكود
st.set_page_config(page_title="CardioDiag", page_icon="🫀", layout="wide")

# ==============================
# CHARGEMENT DU MODÈLE
# ==============================
@st.cache_resource
def load_model():
    model_path = "model.keras"
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            from tensorflow.keras import mixed_precision
            
            # السماح بالدقة المختلطة لأن نموذجك تم تدريبه بها (كما ظهر في الخطأ)
            mixed_precision.set_global_policy('mixed_float16')
            
            # التحميل مع تعطيل الـ compile لتجنب مشاكل الطبقات غير المعروفة
            model = tf.keras.models.load_model(model_path, compile=False)
            return model, True
        except Exception as e:
            st.error(f"Erreur de structure : {e}")
            return None, False
    return None, False

model, is_loaded = load_model()

# ==============================
# PRÉTRAITEMENT
# ==============================
def preprocess_image(img_array):
    # تحجيم الصورة إلى 512x512 (كما في التدريب)
    img = cv2.resize(img_array, (512, 512))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # تحسين التباين CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # التطبيع
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1)) # Shape (1, 512, 512, 1)
    return img

# ==============================
# INTERFACE
# ==============================
st.title("🫀 CardioDiag - Diagnostic IA")

with st.sidebar:
    st.header("⚙️ Paramètres")
    threshold = st.slider("Seuil de décision", 0.0, 1.0, 0.5, 0.01)
    if is_loaded:
        st.success("✅ Modèle réel chargé")
    else:
        st.warning("⚠️ Mode simulation (Fichier non trouvé)")

uploaded_file = st.file_uploader("📂 Charger une image IRM", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)
    
    processed = preprocess_image(original)
    
    if is_loaded:
        # التنبؤ الحقيقي
        prediction = model.predict(processed, verbose=0)
        proba = float(prediction[0][0])
    else:
        # قيمة عشوائية للمحاكاة في حال فشل التحميل
        import random
        proba = random.uniform(0.1, 0.9)

    # عرض النتائج
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original, caption="Image Originale", use_container_width=True)
        
    with col2:
        st.subheader("Résultat")
        diagnosis = "CAD (Malade)" if proba >= threshold else "Normal (Sain)"
        st.metric("Probabilité CAD", f"{proba:.2%}")
        
        if proba >= threshold:
            st.error(f"🚨 Diagnostic : {diagnosis}")
        else:
            st.success(f"✅ Diagnostic : {diagnosis}")
