import streamlit as st
from pinecone import Pinecone
import google.generativeai as genai
from google.generativeai.types import RequestOptions

# 1. IDENTIDAD
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")

with st.sidebar:
    st.header("Guía de Consulta")
    st.markdown("Consultando: **Galaxia de Datos**")
    st.divider()
    st.caption("© 2026 Rubio Intelligence Systems")

st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Doctorando Carlos Rubio")

# 2. CONEXIÓN (FORZANDO VERSIÓN ESTABLE)
try:
    # Pinecone para vectores (Evitamos pedirle vectores a Google para esquivar el 404)
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    
    # Configuración de Gemini con opción de reintento y versión forzada
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Error de inicio: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# 3. LÓGICA DE PROCESAMIENTO
if prompt := st.chat_input("Escriba su consulta legal..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # USAMOS EL EMBEDDING DE PINECONE (Esto es lo que arregla el 404)
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            vector = res_embed[0].values

            # Búsqueda en Galaxia de Datos
            search_results = index.query(vector=vector, top_k=3, include_metadata=True, namespace="silc-juridico")
            context = "\n".join([r['metadata']['text'] for r in search_results['matches']])

            # Respuesta de Gemini (Forzando la API estable v1)
            full_prompt = f"Como experto de Rubio Intelligence Systems, usa este contexto legal: {context}\nPregunta: {prompt}"
            
            # Usamos RequestOptions para asegurar la ruta correcta
            response = model.generate_content(
                full_prompt,
                request_options=RequestOptions(retry=None)
            )
            
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Error técnico: {e}. Intente refrescar la página.")
