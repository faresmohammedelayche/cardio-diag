import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import random

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="CardioDiag", page_icon="🫀", layout="wide")

# ==============================
# PREPROCESSING (MATCH TRAINING)
# ==============================
def custom_preprocess(img_array):
    # تحويل الصورة إلى 512x512 كما فعلت في تدريب النموذج
    img = cv2.resize(img_array, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    # تحويل لرمادي (Grayscale)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # تحسين الصورة (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # إزالة الضوضاء (Blur)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # التطبيع (Normalization)
    normalized = blurred.astype(np.float32) / 255.0
    
    # إعادة التشكيل ليتوافق مع دخل النموذج (1, 512, 512, 1)
    normalized = np.expand_dims(normalized, axis=(0, -1))
    return normalized

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    # جرب الأسماء المحتملة للملف لضمان التحميل
    possible_names = ["model.keras", "cad_cardiac_mri_model (1).keras"]
    model_path = None
    
    for name in possible_names:
        if os.path.exists(name):
            model_path = name
            break

    if model_path:
        try:
            import tensorflow as tf
            # تحميل بدون عمل compile لتجنب مشاكل الإصدارات
            model = tf.keras.models.load_model(model_path, compile=False)
            return model, True
        except Exception as e:
            st.error(f"خطأ في تحميل النموذج: {e}")
            return None, False
    return None, False

model, is_real = load_model()

# ==============================
# MAIN UI
# ==============================
st.title("🫀 CardioDiag - Diagnostic IA")

with st.sidebar:
    st.header("⚙️ Paramètres")
    threshold = st.slider("Seuil de décision", 0.0, 1.0, 0.5, 0.01)
    st.markdown("---")
    if is_real:
        st.success("✅ Modèle réel chargé")
    else:
        st.warning("⚠️ Mode simulation (Fichier non trouvé)")

uploaded_file = st.file_uploader("📂 Charger une image IRM", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)
    
    # المعالجة والتنبؤ
    processed = custom_preprocess(original)
    
    if is_real:
        proba = float(model.predict(processed, verbose=0)[0][0])
    else:
        # إذا فشل التحميل، نستخدم قيمة عشوائية (للعرض فقط)
        proba = random.uniform(0.45, 0.65)

    # منطق التشخيص
    is_cad = proba >= threshold
    diagnosis = "CAD (Malade)" if is_cad else "Normal (Sain)"
    
    # العرض
    st.markdown("---")
    col_img, col_res = st.columns([1, 1])
    
    with col_img:
        st.image(original, caption="Image Originale", use_container_width=True)
        
    with col_res:
        st.subheader("Résultat du Diagnostic")
        st.metric("Probabilité CAD", f"{proba:.2%}")
        st.progress(proba)
        
        if is_cad:
            st.error(f"🚨 Diagnostic : {diagnosis}")
            st.write("⚠️ توصية: يرجى استشارة طبيب مختص فوراً.")
        else:
            st.success(f"✅ Diagnostic : {diagnosis}")
            st.write("🍀 النتيجة تظهر خصائص طبيعية ضمن حدود العتبة المختارة.")
