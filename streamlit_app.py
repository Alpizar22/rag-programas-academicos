import streamlit as st

st.title("🎓 Buscador de Programas Académicos")
st.write("Haz una pregunta sobre un programa y te responderé con base en el brochure.")

pregunta = st.text_input("Escribe tu pregunta aquí")

if pregunta:
    st.write("🧠 Procesando pregunta...")

    # Aquí iría tu lógica de respuesta con RAG
    respuesta = "Esta es una respuesta simulada mientras conectamos el motor RAG."
    st.success(respuesta)
