import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")

# INTERFAZ PROFESIONAL - GRADO ACTUALIZADO
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Doctorando Carlos Rubio")

# CONFIGURACIÓN DE LLAVES (SECRETS)
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# CONFIGURACIÓN DE MODELOS
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# INICIALIZACIÓN DE RECURSOS (CONEXIÓN A GALAXIA-DE-DATOS)
@st.cache_resource
def init_resources():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # CORRECCIÓN: Nombre exacto del índice según tu captura de Pinecone
    index = pc.Index("galaxia-de-datos") 
    # Usamos el modelo que coincide con tus dimensiones de vector
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return index, embed_model

try:
    index, embed_model = init_resources()
except Exception as e:
    st.error(f"Error de conexión con la base de datos legal: {e}")

# INTERFAZ DE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Consulta la legislación mexicana..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. Generar embedding
            query_vector = embed_model.encode(prompt).tolist()
            
            # 2. Buscar en el Namespace 'silc-juridico' que aparece en tu captura
            results = index.query(
                vector=query_vector, 
                top_k=5, 
                include_metadata=True,
                namespace="silc-juridico"
            )
            
            contexto_legal = "\n".join([res['metadata']['text'] for res in results['matches']])

            # 3. Prompt con Identidad Institucional
            full_prompt = f"""
            Eres el SILC, una IA experta en derecho mexicano de Rubio Intelligence Systems.
            Responde de forma técnica y profesional usando este contexto:
            
            {contexto_legal}
            
            Pregunta: {prompt}
            """

            # 4. Generar respuesta
            response = model.generate_content(full_prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"Error al consultar Rubio Intelligence Systems: {e}")
