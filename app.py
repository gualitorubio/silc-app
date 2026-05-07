import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")

# ESTILO Y TÍTULO ACADÉMICO CORREGIDO
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Doctorando Carlos Rubio")

# CONFIGURACIÓN DE LLAVES (SECRETS)
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# CONFIGURACIÓN DE MODELOS
genai.configure(api_key=GEMINI_API_KEY)
# Usamos 'gemini-1.5-flash-latest' para asegurar compatibilidad y evitar el error 404
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# INICIALIZACIÓN DE PINECONE Y EMBEDDINGS
@st.cache_resource
def init_resources():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # IMPORTANTE: Verifica que el nombre de tu índice en Pinecone coincida aquí
    index = pc.Index("leyes-mexico") 
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

if prompt := st.chat_input("Consulta la legislación mexicana (ej. Ley de Expropiación 1936)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. Generar embedding de la consulta
            query_vector = embed_model.encode(prompt).tolist()
            
            # 2. Buscar en Pinecone
            results = index.query(vector=query_vector, top_k=5, include_metadata=True)
            contexto_legal = "\n".join([res['metadata']['text'] for res in results['matches']])

            # 3. Construir el prompt institucional
            full_prompt = f"""
            Actúa como un experto en Derecho Mexicano de Rubio Intelligence Systems.
            Utiliza el siguiente contexto legal para responder la duda del usuario.
            Si el contexto no contiene la respuesta, utiliza tu conocimiento jurídico base.
            
            Contexto recuperado:
            {contexto_legal}
            
            Pregunta del usuario:
            {prompt}
            """

            # 4. Generar respuesta con Gemini
            response = model.generate_content(full_prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"Hubo un problema al procesar la consulta: {e}")
