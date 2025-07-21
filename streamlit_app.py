import streamlit as st
import pandas as pd
import faiss
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

# === Cargar API Key ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 1. Cargar datos ===
df_prog = pd.read_excel("Programas1.xlsx")
df_broch = pd.read_csv("brochures_extraidos.csv")

# Asegurar nombre de columna para hacer merge
df_broch.rename(columns={"Programa": "Nombre de Programa"}, inplace=True)

# === 2. Combinar datasets ===
df = pd.merge(df_prog, df_broch, on="Nombre de Programa", how="left")

# Crear columna de texto combinado
df["texto_completo"] = (
    df["Nombre de Programa"].fillna("") + " | Modalidad: " + df["Modalidad"].fillna("") +
    " | Unidad: " + df["Unidad de Negocio"].fillna("") +
    " | " + df["Texto Brochure"].fillna("")
)

# === 3. Crear embeddings ===
@st.cache_resource(show_spinner="Cargando embeddings...")
def construir_indice(df):
    def get_embedding(texto):
        response = client.embeddings.create(
            input=texto,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    textos = []
    vectores = []
    for texto in df["texto_completo"].dropna().unique():
        try:
            emb = get_embedding(texto)
            textos.append(texto)
            vectores.append(emb)
        except Exception as e:
            print(f"❌ Error con texto: {texto[:30]}... | {e}")

    dim = len(vectores[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectores).astype("float32"))

    return index, textos

index, textos = construir_indice(df)

# === 4. Función para buscar contexto ===
def buscar_contexto(pregunta, k=2):
    def get_embedding(texto):
        response = client.embeddings.create(
            input=texto,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    query_emb = get_embedding(pregunta)
    D, I = index.search(np.array([query_emb]).astype("float32"), k)
    return [textos[i] for i in I[0]]

# === 5. Generar respuesta con contexto ===
def responder_con_contexto(pregunta):
    contexto = buscar_contexto(pregunta)
    prompt = f"""
Usa la siguiente información sobre programas académicos para responder como si fueras un asesor universitario claro, amable y directo. Si no tienes suficiente información, responde con honestidad.

--- CONTEXTO ---
{chr(10).join(contexto)}

--- PREGUNTA ---
{pregunta}

--- RESPUESTA ---
"""
    respuesta = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return respuesta.choices[0].message.content.strip()

# === 6. Interfaz en Streamlit ===
st.title("🎓 Buscador de Programas Académicos")
st.write("Haz una pregunta sobre un programa académico y te responderé con base en el catálogo y brochures institucionales.")

pregunta = st.text_input("✍️ Escribe tu pregunta aquí:")

if pregunta:
    with st.spinner("Buscando respuesta..."):
        respuesta = responder_con_contexto(pregunta)
        st.success(respuesta)
