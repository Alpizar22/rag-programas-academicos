import streamlit as st
import pandas as pd
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import os

# === Cargar datos ===
brochures = pd.read_csv("brochures_extraidos.csv")
perfil_df = pd.read_excel("perfil_completo_sibila_resumen.xlsx", sheet_name="Descripción del perfil")

# === Inicializar modelo de embeddings ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Embeddings del brochure ===
brochures = brochures.dropna(subset=["Texto Brochure"])
brochures["embedding"] = brochures["Texto Brochure"].apply(lambda x: embedder.encode(x, convert_to_tensor=False))

# === Sinónimos para expansión de preguntas ===
sinonimos = {
    "materias": ["asignaturas", "temario", "contenidos", "plan de estudios"],
    "duración": ["cuánto dura", "tiempo de estudio", "años del programa"],
    "costo": ["precio", "valor", "cuánto cuesta", "tarifa"],
    "perfil": ["tipo de estudiante", "características del aspirante", "perfil del alumno"],
}

def expandir_pregunta(pregunta):
    preguntas = [pregunta]
    for palabra, variantes in sinonimos.items():
        if palabra in pregunta.lower():
            preguntas += [pregunta.lower().replace(palabra, alt) for alt in variantes]
    return preguntas

# === Función para obtener tokens ===
def contar_tokens(texto, modelo="gpt-4"):
    encoding = tiktoken.encoding_for_model(modelo)
    return len(encoding.encode(texto))

# === Configuración OpenAI ===
openai.api_key = st.secrets["openai_api_key"]

# === Historial de conversación ===
if "historial" not in st.session_state:
    st.session_state.historial = []

# === Interfaz ===
st.title("🔍 Asistente de Programas Académicos")
pregunta = st.text_input("¿Qué quieres saber sobre un programa académico?")

if pregunta:
    st.session_state.historial.append({"role": "user", "content": pregunta})
    
    pregunta_embedding = embedder.encode(pregunta, convert_to_tensor=False)
    brochures["score"] = brochures["embedding"].apply(lambda x: cosine_similarity([x], [pregunta_embedding])[0][0])
    top = brochures.sort_values("score", ascending=False).iloc[0]

    # Limitar el contexto a 500 tokens
    contexto = top["Texto Brochure"]
    while contar_tokens(contexto) > 500:
        contexto = contexto[:len(contexto) - 100]

    # Construir prompt con historial
    mensajes = st.session_state.historial.copy()
    mensajes.insert(0, {"role": "system", "content": "Eres un asistente experto en programas académicos. Usa el contexto proporcionado para responder con claridad y precisión. Si no hay información relevante, responde solo si el programa es 'Analítica de Negocios', usando el perfil típico como sugerencia secundaria."})
    mensajes.append({"role": "user", "content": f"CONTEXTO:\n{contexto}\n\nPregunta: {pregunta}\nResponde en español, con máximo 600 tokens."})

    respuesta = openai.ChatCompletion.create(
        model="gpt-4",
        messages=mensajes,
        max_tokens=600
    )["choices"][0]["message"]["content"]

    # Si la respuesta es irrelevante y es Analítica de Negocios, usar perfil interpretativo como respaldo
    if ("no tengo" in respuesta.lower() or len(respuesta.strip()) < 50) and "analítica de negocios" in pregunta.lower():
        respuesta = perfil_df.iloc[0, 0]

    st.session_state.historial.append({"role": "assistant", "content": respuesta})

    st.markdown("### 📘 Respuesta:")
    st.write(respuesta)

    st.markdown("---")
    st.markdown(f"🔎 Contexto usado (truncado):\n\n{contexto[:1000]}...")

