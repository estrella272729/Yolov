import cv2
import streamlit as st
import numpy as np
import easyocr

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Placas (Inglés)",
    page_icon="🚗",
    layout="wide"
)

# --- ESTILO VISUAL ---
st.markdown("""
    <style>
        body {
            background-color: #1a1a1a;
            color: #f1f1f1;
        }
        .main {
            background-color: #1a1a1a;
        }
        .title {
            text-align: center;
            color: #FFD60A;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 0.3em;
        }
        .subtitle {
            text-align: center;
            color: #ccc;
            font-size: 1.1em;
            margin-bottom: 1.5em;
        }
        .stButton > button {
            background-color: #FFD60A;
            color: #1a1a1a;
            border: none;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #ffdf40;
        }
        h3, h2 {
            color: #FFD60A;
        }
    </style>
""", unsafe_allow_html=True)

# --- TÍTULO Y DESCRIPCIÓN ---
st.markdown("<div class='title'>Analizador de Placas</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Captura una imagen y el sistema reconocerá la placa del vehículo usando inteligencia artificial.</div>", unsafe_allow_html=True)

# --- CARGA DEL MODELO OCR ---
@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['en', 'es'])  # Reconoce en inglés y español

reader = load_ocr_model()

# --- CAPTURA DE IMAGEN ---
st.markdown("### 📸 Captura o sube una imagen del vehículo")
option = st.radio("Selecciona el método de entrada:", ("Tomar foto", "Subir imagen"))

if option == "Tomar foto":
    picture = st.camera_input("Usa la cámara para tomar la foto")
else:
    picture = st.file_uploader("Sube una imagen del vehículo", type=["jpg", "jpeg", "png"])

# --- PROCESAMIENTO ---
if picture:
    bytes_data = picture.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.image(img, caption="Imagen cargada", use_container_width=True)

    with st.spinner("Analizando imagen y buscando placa..."):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray)

    # --- FILTRAR RESULTADOS POSIBLES DE PLACA ---
    posibles_placas = []
    for (bbox, text, prob) in results:
        if 5 <= len(text) <= 10 and any(char.isdigit() for char in text):  # Placas suelen tener entre 5–10 caracteres
            posibles_placas.append((text, prob))

    st.markdown("### 🧾 Resultado del reconocimiento")
    if posibles_placas:
        best_plate = max(posibles_placas, key=lambda x: x[1])
        st.success(f"Placa detectada: **{best_plate[0]}** (Confianza: {best_plate[1]:.2f})")
    else:
        st.warning("No se detectó ninguna placa con suficiente confianza. Intenta otra imagen o un mejor ángulo.")

# --- PIE DE PÁGINA ---
st.markdown("---")
st.caption("""
**Analizador de Placas – Reconocimiento Automático de Matrículas (LPR)**  
Desarrollado con Streamlit, OpenCV y EasyOCR. Ideal para proyectos de control vehicular o investigación en visión artificial.
""")
