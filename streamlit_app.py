import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Kalp Hastalığı Risk Analizi",
    page_icon="❤️",
    layout="wide"
)

# --- GÜVENLİ IMPORT SİSTEMİ ---
TF_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    pass

st.title("❤️ Kalp Hastalığı Risk Analizi")
st.markdown("---")

# --- MODEL YOLLARI ---
BASE_DIR = os.path.dirname(__file__)
ML_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
DL_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dl_model.keras')

@st.cache_resource
def load_all_assets():
    ml_pipe = None
    dl_mod = None
    
    if os.path.exists(ML_MODEL_PATH):
        ml_pipe = joblib.load(ML_MODEL_PATH)
    
    if TF_AVAILABLE and os.path.exists(DL_MODEL_PATH):
        try:
            dl_mod = tf.keras.models.load_model(DL_MODEL_PATH)
        except:
            pass
            
    return ml_pipe, dl_mod

ml_pipeline, dl_model = load_all_assets()

# --- SIDEBAR GİRİŞLERİ ---
st.sidebar.header("📋 Hasta Bilgileri")

def user_input_features():
    # Numeric
    age = st.sidebar.number_input("Yaş", 20, 90, 50)
    trestbps = st.sidebar.number_input("İstirahat Kan Basıncı (mm Hg)", 90, 200, 120)
    chol = st.sidebar.number_input("Kolesterol (mg/dl)", 100, 600, 200)
    thalach = st.sidebar.number_input("Maksimum Kalp Atış Hızı", 70, 220, 150)
    oldpeak = st.sidebar.number_input("ST Depresyonu (Oldpeak)", 0.0, 6.2, 1.0)
    
    # Categorical
    sex = st.sidebar.selectbox("Cinsiyet", [0, 1], format_func=lambda x: "Kadın" if x == 0 else "Erkek")
    cp = st.sidebar.selectbox("Göğüs Ağrısı Tipi (CP)", [0, 1, 2, 3], 
                              format_func=lambda x: ["Tipik Angina", "Atipik Angina", "Anjinal Olmayan", "Asemptomatik"][x])
    fbs = st.sidebar.selectbox("Açlık Kan Şekeri > 120 mg/dl", [0, 1], format_func=lambda x: "Hayır" if x == 0 else "Evet")
    restecg = st.sidebar.selectbox("İstirahat EKG Sonucu", [0, 1, 2], 
                                   format_func=lambda x: ["Normal", "ST-T Anormalliği", "Hipertrofi"][x])
    exang = st.sidebar.selectbox("Egzersize Bağlı Angina", [0, 1], format_func=lambda x: "Hayır" if x == 0 else "Evet")
    slope = st.sidebar.selectbox("ST Segment Eğimi", [0, 1, 2], 
                                 format_func=lambda x: ["Yukarı Eğimli", "Düz", "Aşağı Eğimli"][x])
    ca = st.sidebar.selectbox("Floroskopi ile Boyanan Damar Sayısı (0-4)", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Talasemi Tipi", [0, 1, 2, 3])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("🧐 İncelenen Hasta Verisi")
st.dataframe(input_df)

if st.button("🚀 Risk Analizi Yap"):
    if ml_pipeline is None:
        st.error("ML Modeli bulunamadı!")
    else:
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        # --- ML Prediction ---
        with c1:
            st.info("🤖 ML Tahmini (Random Forest)")
            try:
                # Predict returns 0 or 1
                pred = ml_pipeline.predict(input_df)[0]
                proba = ml_pipeline.predict_proba(input_df)[0][1] # Probability of 1 (Disease)
                
                res_pct = "{:.1f}".format(proba * 100)
                if pred == 1:
                    st.error(f"**Yüksek Risk!** (Olasılık: %{res_pct})")
                else:
                    st.success(f"**Düşük Risk** (Olasılık: %{res_pct})")
            except Exception as e:
                st.error("Hata: " + str(e))
            
        # --- DL Prediction ---
        with c2:
            st.info("🧠 DL Tahmini (Neural Network)")
            if dl_model:
                try:
                    # Preprocessing using ML pipeline's preprocessor
                    prep_step = ml_pipeline.named_steps.get('preprocessor')
                    if prep_step:
                        X_dl = prep_step.transform(input_df)
                        if hasattr(X_dl, "toarray"): X_dl = X_dl.toarray()
                        
                        # DL output is probability (0-1)
                        pred_dl_prob = dl_model.predict(X_dl, verbose=0)[0][0]
                        pred_dl_class = 1 if pred_dl_prob > 0.5 else 0
                        
                        res_dl_pct = "{:.1f}".format(pred_dl_prob * 100)
                        if pred_dl_class == 1:
                            st.error(f"**Yüksek Risk!** (Olasılık: %{res_dl_pct})")
                        else:
                            st.success(f"**Düşük Risk** (Olasılık: %{res_dl_pct})")
                    else:
                        st.error("Pipeline hatası.")
                except Exception as e:
                    st.warning("DL Tahmin Hatası: " + str(e))
            else:
                st.warning("DL Modeli Aktif Değil")
