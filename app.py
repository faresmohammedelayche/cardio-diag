import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision

st.set_page_config(page_title="CardioDiag", page_icon="🫀", layout="wide")

# ==============================
# CHARGEMENT DU MODÈLE (STRICT)
# ==============================
@st.cache_resource
def load_model_strict():
    model_path = "model.keras"
    if not os.path.exists(model_path):
        st.error(f"❌ Erreur : Le fichier {model_path} est introuvable sur le serveur.")
        st.stop() # إيقاف التطبيق تماماً إذا لم يجد الملف
    
    try:
        # إعدادات التوافق مع النموذج الخاص بك بناءً على الأخطاء السابقة
        mixed_precision.set_global_policy('mixed_float16')
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {e}")
        st.info("نصيحة: تأكد من رفع ملف model.keras مرة أخرى، ربما حدث خطأ أثناء الرفع.")
        st.stop()

# تحميل النموذج إجبارياً
model = load_model_strict()

# ==============================
# PRÉTRAITEMENT
# ==============================
def preprocess_image(img_array):
    # الحجم 225x225 بناءً على معطيات نموذجك
    img = cv2.resize(img_array, (225, 225)) 
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

# ==============================
# INTERFACE
# ==============================
st.title("🫀 CardioDiag - Diagnostic IA")

with st.sidebar:
    st.header("⚙️ Paramètres")
    threshold = st.slider("Seuil de décision", 0.0, 1.0, 0.5, 0.01)
    st.success("✅ Modèle réel opérationnel")

uploaded_file = st.file_uploader("📂 Charger une image IRM", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)
    
    # المعالجة والتنبؤ الحقيقي فقط
    with st.spinner('Analyse en cours...'):
        processed = preprocess_image(original)
        prediction = model.predict(processed, verbose=0)
        proba = float(prediction[0][0])

    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original, caption="Image Originale", use_container_width=True)
        
    with col2:
        st.subheader("Résultat du diagnostic")
        diagnosis = "CAD (Maladie Coronaire)" if proba >= threshold else "Normal (Sain)"
        
        # عرض العداد البياني
        st.metric("Probabilité de pathologie", f"{proba:.2%}")
        
        if proba >= threshold:
            st.error(f"🚨 Résultat : {diagnosis}")
            st.warning("Consultation médicale recommandée.")
        else:
            st.success(f"✅ Résultat : {diagnosis}")
            st.info("L'image présente des caractéristiques normales.")
