import streamlit as st
from pinecone import Pinecone
import google.generativeai as genai

# CONFIGURACIÓN E IDENTIDAD
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️", layout="wide")

# Barra lateral (Garantizada)
with st.sidebar:
    st.header("Guía de Consulta")
    st.markdown("1. Ingrese su duda jurídica.\n2. Búsqueda en **Galaxia de Datos** (1024 dim).")
    st.divider()
    st.caption("© 2026 Rubio Intelligence Systems | Doctorando Carlos Rubio")

st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Doctorando Carlos Rubio")

# INICIALIZACIÓN
try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Error de configuración: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "SILC en línea. ¿Qué análisis legal realizaremos hoy?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# PROCESAMIENTO
if prompt := st.chat_input("Consulta legal..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # SOLUCIÓN MAESTRA: Usar la inferencia de Pinecone para evitar el 404 de Google
            embeddings = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            query_vector = embeddings[0].values

            results = index.query(
                vector=query_vector,
                top_k=5,
                include_metadata=True,
                namespace="silc-juridico"
            )

            contexto = "\n".join([res['metadata']['text'] for res in results['matches']])
            full_prompt = f"Eres el SILC de Rubio Intelligence Systems. Contexto: {contexto}\nPregunta: {prompt}"
            
            response = model.generate_content(full_prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Error técnico: {e}")
