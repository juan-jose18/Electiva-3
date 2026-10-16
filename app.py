from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "modelos_plantvillage"

MODELOS_DISPONIBLES = {
    "ResNet50V2 (mejor precisión)": MODELS_DIR / "ResNet50V2_best.keras",
    "MobileNetV2 (rápido)": MODELS_DIR / "MobileNetV2_best.keras",
    "Student KD (destilado)": MODELS_DIR / "Student_KD_final.keras",
}

IMG_SIZE = (224, 224)
MIN_GREEN_RATIO = 0.03

# ---------------------------------------------------------------------------
# Nombres en español
# ---------------------------------------------------------------------------
NOMBRES_ES = {
    "Adelfilla": ("Adelfilla", "Sana"),
    "Amapola": ("Amapola", "Sana"),
    "Cichorium": ("Achicoria", "Sana"),
    "Digitas Purpurea": ("Digitalis Púrpura", "Sana"),
    "Helecho": ("Helecho", "Sano"),
    "Hiedra Venenosa": ("Hiedra Venenosa", "Sana"),
    "Laurel Montaña": ("Laurel de Montaña", "Sano"),
    "Linaria Vulgari": ("Linaria Vulgar", "Sana"),
    "Margarita": ("Margarita", "Sana"),
    "Muguete": ("Muguete", "Sano"),
    "Pepper__bell___Bacterial_spot": ("Pimiento", "Mancha Bacteriana"),
    "Pepper__bell___healthy": ("Pimiento", "Sano"),
    "Potato___Early_blight": ("Papa", "Tizón Temprano"),
    "Potato___Late_blight": ("Papa", "Tizón Tardío"),
    "Potato___healthy": ("Papa", "Sana"),
    "Tomato_Bacterial_spot": ("Tomate", "Mancha Bacteriana"),
    "Tomato_Early_blight": ("Tomate", "Tizón Temprano"),
    "Tomato_Late_blight": ("Tomate", "Tizón Tardío"),
    "Tomato_Leaf_Mold": ("Tomate", "Moho de la Hoja"),
    "Tomato_Septoria_leaf_spot": ("Tomate", "Mancha de Septoria"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("Tomate", "Ácaro Rojo"),
    "Tomato__Target_Spot": ("Tomate", "Mancha Objetivo"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("Tomate", "Virus del Enrollamiento Amarillo"),
    "Tomato__Tomato_mosaic_virus": ("Tomate", "Virus del Mosaico"),
    "Tomato_healthy": ("Tomate", "Sano"),
}


def parsear_clase(nombre_clase: str):
    if nombre_clase in NOMBRES_ES:
        return NOMBRES_ES[nombre_clase]
    return nombre_clase, "Desconocida"


def ratio_region_vegetal(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([15, 30, 30]), np.array([35, 255, 255]))
    mask_brown = cv2.inRange(hsv, np.array([5, 20, 20]), np.array([25, 200, 180]))
    mask = cv2.bitwise_or(mask_green, mask_yellow)
    mask = cv2.bitwise_or(mask, mask_brown)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return float(np.count_nonzero(mask)) / float(mask.size)


def es_posible_hoja(img_bgr: np.ndarray, pred: np.ndarray):
    green_ratio = ratio_region_vegetal(img_bgr)
    confianza = float(np.max(pred) * 100.0)

    if green_ratio < MIN_GREEN_RATIO:
        return False, (
            "La imagen no parece una hoja de planta. "
            "Sube una foto de una hoja (tomate, papa, pimiento u otra planta del sistema)."
        ), green_ratio, confianza

    if confianza < 12.0:
        return False, (
            "Confianza demasiado baja. Prueba con una foto más clara y centrada de la hoja."
        ), green_ratio, confianza

    return True, "", green_ratio, confianza


st.set_page_config(
    page_title="Clasificador de Enfermedades en Hojas",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Rajdhani:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Rajdhani', system-ui, sans-serif; }
.stApp {
    background:
        radial-gradient(ellipse at 20% 0%, rgba(0, 255, 170, 0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(0, 140, 255, 0.10) 0%, transparent 45%),
        linear-gradient(160deg, #05080f 0%, #0a1220 40%, #071018 100%);
    color: #e8fff6;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #03060c 0%, #0a1628 50%, #061820 100%) !important;
    border-right: 1px solid rgba(0, 255, 200, 0.25);
}
section[data-testid="stSidebar"] * { color: #c8ffe9 !important; }
.hero {
    background: linear-gradient(135deg, rgba(0,40,35,0.95) 0%, rgba(0,20,50,0.95) 55%, rgba(10,5,40,0.95) 100%);
    border: 1px solid rgba(0,255,200,0.35); border-radius: 20px;
    padding: 1.5rem 1.7rem; margin-bottom: 1.2rem;
    box-shadow: 0 0 30px rgba(0,255,170,0.15);
}
.hero h1 {
    font-family: 'Orbitron', sans-serif !important; margin: 0 0 0.3rem 0 !important;
    font-size: 1.65rem !important; color: #7dffc8 !important;
}
.hero p { margin: 0 !important; color: #a8d4c8 !important; font-size: 1.02rem !important; }
.neo-card {
    background: rgba(8,18,30,0.85); border: 1px solid rgba(0,255,200,0.22);
    border-radius: 16px; padding: 1.05rem 1.25rem; margin-bottom: 0.75rem;
}
.neo-label {
    font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: #5ee0b5; font-weight: 600; margin-bottom: 0.22rem;
}
.neo-value { font-family: 'Orbitron', sans-serif; font-size: 1.25rem; font-weight: 700; color: #e9fff7; }
.badge-high {
    display: inline-block; padding: 0.32rem 0.85rem; border-radius: 999px; font-weight: 700;
    background: rgba(0,255,160,0.15); color: #7dffc8; border: 1px solid rgba(0,255,180,0.45);
}
.badge-mid {
    display: inline-block; padding: 0.32rem 0.85rem; border-radius: 999px; font-weight: 700;
    background: rgba(255,200,0,0.12); color: #ffd666; border: 1px solid rgba(255,200,0,0.4);
}
.badge-low {
    display: inline-block; padding: 0.32rem 0.85rem; border-radius: 999px; font-weight: 700;
    background: rgba(255,70,100,0.12); color: #ff8aa0; border: 1px solid rgba(255,80,110,0.4);
}
.alert-ok {
    background: rgba(0,255,160,0.08); border: 1px solid rgba(0,255,180,0.35);
    border-left: 4px solid #00e6a0; border-radius: 12px; padding: 0.85rem 1.05rem;
    color: #9fffd8; font-weight: 600;
}
.alert-warn {
    background: rgba(255,180,0,0.08); border: 1px solid rgba(255,190,0,0.35);
    border-left: 4px solid #ffc107; border-radius: 12px; padding: 0.85rem 1.05rem;
    color: #ffe399; font-weight: 600;
}
.alert-reject {
    background: rgba(255,40,80,0.10); border: 1px solid rgba(255,60,100,0.45);
    border-left: 4px solid #ff3366; border-radius: 14px; padding: 1.1rem 1.25rem;
    color: #ffb3c4; font-weight: 600;
}
.alert-reject h3 {
    font-family: 'Orbitron', sans-serif; color: #ff6b8a !important;
    margin: 0 0 0.4rem 0 !important; font-size: 1.1rem !important;
}
.top-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0.8rem; border-radius: 12px;
    background: rgba(10,22,36,0.9); margin-bottom: 0.38rem;
    border: 1px solid rgba(0,255,200,0.15);
}
.top-rank {
    width: 1.55rem; height: 1.55rem; border-radius: 50%;
    background: linear-gradient(135deg, #00c98a, #00a0ff); color: #041018;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 800; margin-right: 0.5rem;
}
.chip {
    display: inline-block; padding: 0.26rem 0.65rem; margin: 0.15rem;
    border-radius: 999px; font-weight: 700; font-size: 0.84rem;
    background: rgba(0,255,200,0.08); border: 1px solid rgba(0,255,200,0.3); color: #7dffc8;
}
.footer {
    margin-top: 1.7rem; padding-top: 0.85rem;
    border-top: 1px solid rgba(0,255,200,0.15);
    color: #6a9a88; font-size: 0.86rem; text-align: center;
}
#MainMenu, footer, header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


def badge_class(confianza: float) -> str:
    if confianza >= 85:
        return "badge-high"
    if confianza >= 60:
        return "badge-mid"
    return "badge-low"


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_rgb, IMG_SIZE)
    x = img.astype("float32") / 255.0
    return np.expand_dims(x, axis=0)


@st.cache_resource
def cargar_modelo(ruta: str):
    return tf.keras.models.load_model(ruta)


@st.cache_data
def cargar_clases():
    path = MODELS_DIR / "class_names.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


with st.sidebar:
    st.markdown("## 🍃 Clasificador Foliar")
    st.markdown("PlantVillage + clases propias")
    st.markdown("---")

    modelos_existentes = {
        nombre: str(ruta)
        for nombre, ruta in MODELOS_DISPONIBLES.items()
        if ruta.exists()
    }

    if not modelos_existentes:
        st.error("No se encontraron modelos en `modelos_plantvillage/`")
        st.stop()

    modelo_elegido = st.selectbox(
        "Modelo a usar",
        options=list(modelos_existentes.keys()),
        index=0,
    )

    st.markdown("---")
    st.markdown("**Cultivos y plantas**")
    st.markdown("🍅 Tomate · 🥔 Papa · 🫑 Pimiento")
    st.markdown("🌸 Adelfilla · Amapola · Achicoria")
    st.markdown("🌿 Helecho · Hiedra · Laurel · etc.")
    st.markdown("---")
    st.caption("UCC · Ibagué–Espinal · 2026")


st.markdown(
    """
<div class="hero">
  <h1>CLASIFICADOR DE ENFERMEDADES EN HOJAS</h1>
  <p>Transfer Learning · Knowledge Distillation · Grad-CAM · 25 clases</p>
</div>
""",
    unsafe_allow_html=True,
)

class_names = cargar_clases()
if class_names is None:
    st.error("No se encontró `modelos_plantvillage/class_names.json`")
    st.stop()

try:
    model = cargar_modelo(modelos_existentes[modelo_elegido])
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

col_up, col_info = st.columns([1.4, 1])

with col_up:
    st.markdown("#### 📡 Sube una imagen de hoja")
    archivo = st.file_uploader(
        "JPG / PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

with col_info:
    st.markdown("#### 🌐 Cobertura")
    st.markdown(
        """
        <div class="neo-card">
          <span class="chip">🍅 Tomate</span>
          <span class="chip">🥔 Papa</span>
          <span class="chip">🫑 Pimiento</span>
          <span class="chip">🌸 Otras plantas</span>
          <p style="margin:0.75rem 0 0 0; color:#8ebdac; font-size:0.93rem;">
            Sube una foto de hoja. Si la imagen no parece vegetal, el sistema la rechazará.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if archivo is None:
    st.info("⬆️ Carga una imagen para iniciar el análisis.")
    st.markdown(
        """
        <div class="footer">
          Herramienta de apoyo · No sustituye diagnóstico fitopatológico profesional<br>
          Ingeniería de Sistemas · Universidad Cooperativa de Colombia · 2026
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("No se pudo leer la imagen.")
    st.stop()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
x = preprocess(img_rgb)

with st.spinner("⚙️ Analizando imagen..."):
    pred = model.predict(x, verbose=0)[0]

idx = int(np.argmax(pred))
confianza = float(pred[idx] * 100.0)
clase_raw = class_names[idx]
planta, enfermedad = parsear_clase(clase_raw)
es_sana = enfermedad.lower() in {"sana", "sano"}

valida, msg_rechazo, green_ratio, conf_chk = es_posible_hoja(img_bgr, pred)

st.markdown("#### 🧪 Resultado del análisis")

col_img, col_res = st.columns([1.05, 1.25], gap="large")

with col_img:
    st.image(img_rgb, caption="Imagen analizada", use_container_width=True)
    st.caption(f"Señal vegetal estimada: **{green_ratio * 100:.1f}%**")

with col_res:
    if not valida:
        st.markdown(
            f"""
            <div class="alert-reject">
              <h3>🚫 IMAGEN NO COMPATIBLE</h3>
              <p style="margin:0;">{msg_rechazo}</p>
              <p style="margin:0.65rem 0 0 0; font-size:0.9rem; opacity:0.9;">
                Confianza máx.: <b>{confianza:.1f}%</b> ·
                Cobertura vegetal: <b>{green_ratio * 100:.1f}%</b>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning("El sistema está entrenado para reconocer hojas de plantas.")
    else:
        st.markdown(
            f"""
            <div class="neo-card">
              <div class="neo-label">Planta detectada</div>
              <div class="neo-value">{planta}</div>
            </div>
            <div class="neo-card">
              <div class="neo-label">Estado</div>
              <div class="neo-value">{"SANA" if es_sana else "ENFERMA / AFECTADA"}</div>
            </div>
            <div class="neo-card">
              <div class="neo-label">Diagnóstico</div>
              <div class="neo-value">{enfermedad}</div>
            </div>
            <div class="neo-card">
              <div class="neo-label">Confianza</div>
              <div><span class="{badge_class(confianza)}">{confianza:.2f} %</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if es_sana:
            st.markdown(
                '<div class="alert-ok">✅ La hoja se clasifica como <b>SANA</b>.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="alert-warn">⚠️ Posible enfermedad: <b>{enfermedad}</b>. '
                "Si el caso es crítico, valide con un técnico agrícola.</div>",
                unsafe_allow_html=True,
            )

        st.progress(min(confianza / 100.0, 1.0))

if valida:
    st.markdown("#### 📊 Top 3 predicciones")
    top3_idx = np.argsort(pred)[::-1][:3]
    rows = []
    for rank, i in enumerate(top3_idx, start=1):
        p_name, e_name = parsear_clase(class_names[i])
        pct = float(pred[i] * 100.0)
        rows.append(
            f"""
            <div class="top-row">
              <div>
                <span class="top-rank">{rank}</span>
                <strong>{p_name}</strong> · {e_name}
              </div>
              <div style="font-weight:700; color:#7dffc8;">{pct:.2f}%</div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer">
      Clasificador de enfermedades en hojas · herramienta de <b>apoyo</b><br>
      ResNet50V2 / MobileNetV2 / Knowledge Distillation · UCC Ibagué–Espinal · 2026
    </div>
    """,
    unsafe_allow_html=True,
)
