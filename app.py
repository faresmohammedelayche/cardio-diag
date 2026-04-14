# ==============================
# app.py - Interface IHM professionnelle
# Diagnostic IRM cardiaque (CAD vs Normal)
# ==============================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random
import os
import time

# ==============================
# CONFIGURATION DE LA PAGE
# ==============================
st.set_page_config(
    page_title="CardioDiag - IRM Cardiaque",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CSS PERSONNALISÉ POUR UN STYLE PRO
# ==============================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .diagnostic-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-bar {
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #95a5a6;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# PRÉTRAITEMENT (identique à l'entraînement)
# ==============================
def custom_preprocess(img_array):
    """
    CLAHE + GaussianBlur + normalisation [0,1]
    Entrée : uint8 RGB
    Sortie : float32 RGB normalisé
    """
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
# CHARGEMENT DU MODÈLE (réel ou factice)
# ==============================
@st.cache_resource
def load_model():
    model_path = "model.h5"  # ou .h5
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            return model, True
        except Exception as e:
            st.warning(f"⚠️ Impossible de charger le modèle réel : {e}")
            return None, False
    else:
        st.info("ℹ️ Modèle réel non trouvé. Utilisation d'un modèle factice (prédictions aléatoires).")
        return None, False

model, is_real = load_model()

def predict(processed_img):
    """Prédiction réelle ou factice"""
    if is_real:
        import tensorflow as tf
        input_tensor = np.expand_dims(processed_img, axis=0)
        proba = float(model.predict(input_tensor, verbose=0)[0][0])
    else:
        # Simulation réaliste : tendance à varier autour de 0.5
        proba = random.uniform(0.2, 0.8)
    return proba

# ==============================
# SIDEBAR : PARAMÈTRES ET VALIDATION
# ==============================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=80)
    st.title("⚙️ Paramètres")
    
    threshold = st.slider(
        "Seuil de décision",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Si probabilité ≥ seuil → diagnostic CAD"
    )
    
    st.markdown("---")
    st.subheader("🔍 Validation (optionnelle)")
    true_label = st.radio(
        "Classe réelle de l'image (pour évaluation)",
        options=["Non renseignée", "Normal", "CAD"],
        index=0,
        help="Indiquez la véritable classe pour vérifier la prédiction"
    )
    
    st.markdown("---")
    st.subheader("ℹ️ Information")
    if is_real:
        st.success("✅ Modèle réel chargé")
    else:
        st.warning("⚠️ Mode factice actif")
    st.caption("Modèle CNN entraîné sur le dataset CAD-Cardiac-MRI")

# ==============================
# ZONE PRINCIPALE : TITRE ET CHARGEMENT
# ==============================
st.markdown('<div class="main-header">🫀 CardioDiag - IRM Cardiaque</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Diagnostic assisté par intelligence artificielle</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1])

with col_left:
    uploaded_file = st.file_uploader(
        "📂 Charger une image IRM",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Formats supportés : PNG, JPG, JPEG, BMP, TIFF"
    )

# ==============================
# TRAITEMENT DE L'IMAGE
# ==============================
if uploaded_file is not None:
    # Lecture
    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)
    
    # Prétraitement
    with st.spinner("Prétraitement de l'image en cours..."):
        processed = custom_preprocess(original)
        time.sleep(0.5)  # Pour l'effet visuel
    
    # Affichage côte à côte
    with col_right:
        st.markdown("### 🖼️ Visualisation du prétraitement")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(original, caption="Image originale", use_container_width=True)
        with col_img2:
            st.image(processed, caption="Après CLAHE + flou", use_container_width=True)
    
    # Prédiction
    proba = predict(processed)
    diagnosis = "CAD (maladie coronarienne)" if proba >= threshold else "Normal"
    confidence = proba if diagnosis == "CAD" else 1 - proba
    
    # Affichage des résultats
    st.markdown("---")
    st.markdown("## 📊 Résultat du diagnostic")
    
    # Métriques en colonnes
    col_met1, col_met2, col_met3 = st.columns(3)
    with col_met1:
        st.metric("Diagnostic", diagnosis, delta=None)
    with col_met2:
        st.metric("Probabilité CAD", f"{proba:.2%}", delta=None)
    with col_met3:
        st.metric("Niveau de confiance", f"{confidence:.2%}", delta=None)
    
    # Barre de probabilité stylisée
    st.markdown('<div class="confidence-bar">', unsafe_allow_html=True)
    st.progress(proba)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Message d'alerte ou succès
    if diagnosis == "CAD":
        st.warning("⚠️ **Recommandation** : Une consultation médicale spécialisée est conseillée.")
    else:
        st.success("✅ **Conclusion** : L'image présente des caractéristiques normales.")
    
    # ==============================
    # VALIDATION AVEC CLASSE RÉELLE
    # ==============================
    if true_label != "Non renseignée":
        st.markdown("---")
        st.subheader("🔍 Évaluation de la prédiction")
        expected = "Normal" if true_label == "Normal" else "CAD"
        is_correct = (diagnosis == expected)
        
        if is_correct:
            st.success(f"✅ **Prédiction correcte** ! La classe réelle est **{expected}**.")
        else:
            st.error(f"❌ **Prédiction incorrecte**. La classe réelle est **{expected}** alors que le modèle a prédit **{diagnosis}**.")
        
        # Affichage d'une matrice de confusion simplifiée
        st.caption("Utilisez cette information pour évaluer la performance du modèle sur vos propres images.")
    
else:
    with col_right:
        st.info("👈 **Commencez par charger une image IRM**\n\nLes formats supportés : PNG, JPG, JPEG, BMP, TIFF")

# ==============================
# PIED DE PAGE
# ==============================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Modèle CNN entraîné sur le dataset CAD-Cardiac-MRI (Kaggle) | Prétraitement : CLAHE + GaussianBlur</p>
    <p>⚠️ Application à usage éducatif et de recherche. Ne remplace pas un avis médical.</p>
</div>
""", unsafe_allow_html=True)
