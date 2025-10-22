import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="Reconocedor de Personas - Agencia Secreta",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# Encabezado temático
st.markdown("""
<style>
h1 {
    color: #FF007F !important; /* Rosa neón */
    text-align: center;
}
h3, h2 {
    color: #E0E0E0 !important;
}
.reportview-container {
    background-color: #0E1117;
}
.stMarkdown, .stDataFrame {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# Título y descripción de la aplicación
st.title(" Sistema de Reconocimiento Facial - Agencia Secreta")
st.markdown("""
Bienvenido, **agente**.  
Este sistema clasificado de la Agencia analiza rostros y detecta individuos en tiempo real.  
Carga una imagen o utiliza la cámara para escanear posibles **objetivos** en el entorno.
""")

# Función para cargar el modelo YOLOv5
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            model = yolov5.load(model_path)
            return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Verifica dependencias:
        - torch==1.12.0
        - yolov5==7.0.9
        """)
        return None

# Cargar modelo
with st.spinner("Inicializando sistema de visión artificial..."):
    model = load_yolov5_model()

# Parámetros del sistema
if model:
    st.sidebar.title(" Panel de Control - Clasificación de Objetivos")
    st.sidebar.markdown("Ajusta los parámetros de detección del sistema.")
    
    model.conf = st.sidebar.slider('Nivel de confianza mínimo', 0.0, 1.0, 0.25, 0.01)
    model.iou = st.sidebar.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
    model.agnostic = st.sidebar.checkbox('Modo agnóstico', False)
    model.multi_label = st.sidebar.checkbox('Múltiples etiquetas', False)
    model.max_det = st.sidebar.number_input('Máx. detecciones', 10, 2000, 1000, 10)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Solo para uso interno del Departamento de Inteligencia Visual 🕶️")

    # Contenedor principal
    st.markdown("## 📸 Escáner de Rostros en Tiempo Real")
    st.markdown("Activa la cámara o sube una imagen para identificar posibles **agentes o sospechosos**.")
    
    picture = st.camera_input("Captura una imagen para analizar", key="camera_input")

    if picture:
        bytes_data = picture.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Analizando rostros... "):
            try:
                results = model(frame)
            except Exception as e:
                st.error(f"Error durante el análisis: {str(e)}")
                st.stop()
        
        predictions = results.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(" Imagen Procesada - Clasificación Visual")
            results.render()
            st.image(results.ims[0][:, :, ::-1], use_container_width=True)

        with col2:
            st.subheader("📋 Reporte de Objetivos Identificados")
            label_names = model.names
            data = []

            for i in range(len(categories)):
                label = label_names[int(categories[i])]
                confidence = scores[i].item()
                data.append({
                    "Identificación": label.title(),
                    "Confianza del Sistema": f"{confidence:.2f}"
                })
            
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index('Identificación')['Confianza del Sistema'].astype(float))
            else:
                st.info("No se detectaron individuos con los parámetros actuales.")
else:
    st.error("No se pudo iniciar el sistema de reconocimiento. Verifica dependencias y permisos.")

# Pie de página
st.markdown("---")
st.caption("""
🕶️ **Agencia Secreta de Inteligencia Artificial (ASIA)**  
Sistema experimental de reconocimiento facial basado en YOLOv5 + Streamlit  
*Clasificado - Acceso restringido a personal autorizado*
""")
