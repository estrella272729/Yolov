import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
import os
import sys

# Instalar EasyOCR automáticamente si no está presente
os.system("pip install easyocr==1.7.1 torch torchvision --quiet")

import easyocr

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento de Placas Vehiculares",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Reconocimiento de Placas Vehiculares")
st.markdown("""
Esta aplicación utiliza **YOLOv5** para detectar vehículos y **EasyOCR** para reconocer texto en las placas.  
Puedes capturar una imagen con tu cámara o subir una foto de un automóvil.
""")

# Cargar modelo YOLOv5
@st.cache_resource
def load_yolov5_model():
    try:
        import yolov5
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"Error al cargar YOLOv5: {str(e)}")
        return None

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

# Cargar OCR
reader = easyocr.Reader(['en'])

st.sidebar.title("⚙️ Configuración de detección")
conf = st.sidebar.slider("Nivel de confianza mínimo", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)

st.sidebar.markdown("---")
upload_option = st.sidebar.radio("📸 Fuente de imagen", ["Usar cámara", "Subir imagen"])

# Captura o carga de imagen
if upload_option == "Usar cámara":
    picture = st.camera_input("Captura una foto del vehículo")
else:
    picture = st.file_uploader("Sube una imagen del vehículo", type=["jpg", "jpeg", "png"])

# Procesar la imagen
if picture is not None and model:
    bytes_data = picture.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Detección con YOLO
    with st.spinner("Detectando vehículos y placas..."):
        results = model(image)
        results.render()  # Dibuja los rectángulos
    
    # Mostrar imagen con detecciones
    st.image(results.ims[0][:, :, ::-1], caption="Vehículos detectados", use_container_width=True)
    
    # Extraer regiones y aplicar OCR
    st.markdown("### 🔍 Resultados del reconocimiento")
    df_data = []
    
    try:
        for *box, conf_score, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            if label in ["car", "truck", "bus", "motorbike"]:
                x1, y1, x2, y2 = map(int, box)
                crop = image[y1:y2, x1:x2]
                
                # OCR en la región
                ocr_results = reader.readtext(crop)
                
                for (bbox, text, prob) in ocr_results:
                    if len(text) >= 4:  # Filtrar textos cortos
                        df_data.append({
                            "Vehículo": label,
                            "Texto detectado": text,
                            "Confianza OCR": round(prob, 2)
                        })
        
        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No se detectaron placas legibles en la imagen.")
    
    except Exception as e:
        st.error(f"Error en el reconocimiento OCR: {str(e)}")

# Pie de página
st.markdown("---")
st.caption("""
**Desarrollado por:** Sistema de Reconocimiento de Placas Vehiculares  
Basado en YOLOv5 + EasyOCR + Streamlit
""")

