import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from dotenv import load_dotenv
from openai import OpenAI

# === 1. Configuración ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 2. Cargar datos ===
df_prog = pd.read_excel("Programas1.xlsx")
df_broch = pd.read_csv("brochures_extraidos.csv")
df_broch.rename(columns={"Programa": "Nombre de Programa"}, inplace=True)
df = pd.merge(df_prog, df_broch, on="Nombre de Programa", how="left")

# === 3. Crear texto completo ===
df["texto_completo"] = (
    df["Nombre de Programa"].fillna("") + " | Modalidad: " + df["Modalidad"].fillna("") +
    " | Unidad: " + df["Unidad de Negocio"].fillna("") +
    " | " + df["Texto Brochure"].fillna("")
)

# === 4. Embeddings ===
def get_embedding(texto):
    response = client.embeddings.create(
        input=texto,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

textos = df["texto_completo"].dropna().unique().tolist()
vectores = [get_embedding(t) for t in textos]

index = faiss.IndexFlatL2(len(vectores[0]))
index.add(np.array(vectores).astype("float32"))

# === 5. Sinónimos para expansión de preguntas ===
sinonimos = {
    "materias": ["asignaturas", "temario", "contenidos", "plan de estudios"],
    "duración": ["cuánto dura", "tiempo de estudio", "años del programa"],
    "costo": ["precio", "valor", "cuánto cuesta", "tarifa"],
}

def expandir_pregunta(pregunta):
    preguntas = [pregunta]
    for palabra, variantes in sinonimos.items():
        if palabra in pregunta.lower():
            preguntas += [pregunta.lower().replace(palabra, alt) for alt in variantes]
    return preguntas

# === 6. Buscar textos relevantes ===
def buscar_contexto(pregunta, k=3):
    alternativas = expandir_pregunta(pregunta)
    candidatos = []

    for alt in alternativas:
        emb = get_embedding(alt)
        D, I = index.search(np.array([emb]).astype("float32"), k)
        candidatos += [(textos[i], D[0][j]) for j, i in enumerate(I[0])]

    candidatos = sorted(candidatos, key=lambda x: x[1])
    return [c[0] for c in candidatos[:k]]

# === 7. Generar respuesta ===
def responder_con_contexto(pregunta):
    contexto = buscar_contexto(pregunta, k=2)
    prompt = f"""
Responde como un asesor académico profesional y claro. Usa la siguiente información para contestar.
Si no tienes suficiente información, responde con sinceridad.

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

# === 8. Generar perfil interpretativo por aspirante ===
def generar_perfil_aspirante(programa, estado, fase, palabras_clave, motivo_descartado=None):
    pregunta = f"""
Dado que un aspirante está interesado en el programa '{programa}',
vive en {estado}, se encuentra en la fase '{fase}' del proceso
y ha utilizado palabras como: {palabras_clave}.
{f'Se descartó por: {motivo_descartado}.' if motivo_descartado else ''}

Genera un análisis breve y profesional con:
1. Perfil probable del aspirante
2. Posibles riesgos de desalineación con el programa
3. Recomendaciones para seguimiento o reasignación
"""
    contexto = buscar_contexto(programa, k=2)
    prompt = f"""--- CONTEXTO ---\n{chr(10).join(contexto)}\n\n--- PREGUNTA ---\n{pregunta}"""

    respuesta = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return respuesta.choices[0].message.content.strip()

# === 9. Interfaz Streamlit ===
st.title("🎓 Asistente de Programas Académicos")

# Consulta normal
st.markdown("## 🔍 Pregunta sobre un programa")
pregunta = st.text_input("Haz una pregunta sobre un programa académico:")
if pregunta:
    with st.spinner("Buscando la mejor respuesta..."):
        respuesta = responder_con_contexto(pregunta)
    st.markdown("### 🧠 Respuesta:")
    st.write(respuesta)

# Análisis por aspirante
st.markdown("## 👤 Análisis por aspirante")
col1, col2 = st.columns(2)
programa = col1.selectbox("Programa", df["Nombre de Programa"].unique())
estado = col2.text_input("Estado de procedencia")
fase = st.selectbox("Fase del proceso", ["Por contactar", "Interesado", "Indeciso", "Inscrito", "Descarte"])
palabras = st.text_area("Palabras clave del aspirante")
motivo = st.text_input("Motivo de descarte (opcional)")

if st.button("Generar perfil interpretativo"):
    with st.spinner("Generando análisis..."):
        perfil = generar_perfil_aspirante(programa, estado, fase, palabras, motivo)
    st.markdown("### 🧠 Perfil generado:")
    st.write(perfil)

