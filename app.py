import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Cargar modelo y scaler
model = joblib.load("mejor_modelo_xgb.pkl")

# Configuración del layout
st.set_page_config(page_title="Predicción de Riesgo de Default Bancario", layout="wide")

# Título e imagen de bienvenida
st.title("Predicción de Riesgo de Default Bancario")
st.markdown("**Bienvenido a la herramienta de predicción de riesgo de default bancario.**")

# Mostrar imagen relacionada con el tema
image = Image.open("imagen_default.jpg")  # Cambia el nombre por la imagen que desees
st.image(image, caption="Previsión de Riesgo de Default Bancario", use_column_width=True)

# Cargar el dataset y las columnas (si es necesario para mostrar diccionario)
data = pd.read_csv("base10.csv")
columnas = data.drop(columns=["FLG_CLI_DEF60"]).columns

# Si tienes un diccionario de variables, podrías cargarlo aquí
diccionario = pd.read_excel("DICCIONARIO.xlsx", sheet_name="Variables")

# Mostrar el diccionario de variables al usuario
st.sidebar.title("Diccionario de Variables")
for index, row in diccionario.iterrows():
    st.sidebar.markdown(f"**{row['Variable']}**: {row['Descripción']}")

# Formulario de entrada de datos
st.subheader("Ingresa los datos del cliente:")
input_data = {}
for col in columnas:
    input_data[col] = st.number_input(f"{col}", value=float(data[col].mean()), step=0.01)

# Botón de predicción
if st.button("Predecir"):
    input_df = pd.DataFrame([input_data])
    
    # Normalizar los datos con el mismo scaler usado para entrenar el modelo
    scaler = StandardScaler()
    scaler.fit(data.drop(columns=["FLG_CLI_DEF60"]))
    input_scaled = scaler.transform(input_df)
    
    # Predicción
    prediction = model.predict(input_scaled)[0]
    
    # Mostrar resultado
    st.subheader("Resultado de la Predicción:")
    if prediction == 1:
        st.error("⚠️ **Riesgo de Default (más de 60 días de atraso)**")
    else:
        st.success("✅ **No hay riesgo de Default (cliente seguro)**")
    
    # Mostrar gráfico de probabilidad
    prob = model.predict_proba(input_scaled)[0][1]  # Probabilidad de default
    st.write(f"Probabilidad de Default: {prob * 100:.2f}%")
    
    fig, ax = plt.subplots()
    ax.barh(['Default', 'No Default'], [prob, 1 - prob], color=['red', 'green'])
    ax.set_xlim(0, 1)
    st.pyplot(fig)

# Mejorar la interfaz con un pie de página
st.markdown("""
    ---
    #### ¡Gracias por usar nuestra herramienta de predicción!  
    Desarrollado por el equipo de análisis de riesgo bancario.
""")

# Puedes agregar más estilos y personalizaciones de colores con `st.markdown()` y HTML si lo deseas.
