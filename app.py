import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# 1. CONFIGURACIÓN E IDENTIDAD
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️", layout="wide")

st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Doctorando Carlos Rubio")

# 2. BARRA LATERAL
with st.sidebar:
    st.header("Guía de Consulta")
    st.markdown("Consultando la **Galaxia de Datos** (1024 dimensiones).")
    st.divider()
    st.caption("© 2026 Rubio Intelligence Systems")

# 3. CONFIGURACIÓN DE LLAVES (SECRETS)
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# 4. RECURSOS
@st.cache_resource
def init_resources():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index("galaxia-de-datos")

index = init_resources()

# Inicialización de historial
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bienvenido al SILC. La conexión con la Galaxia de Datos es estable. ¿Qué análisis legal desea realizar?"}]

# Mostrar mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. LÓGICA DE PROCESAMIENTO
if prompt := st.chat_input("Introduzca su consulta legal aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando registros legales en Rubio Intelligence Systems..."):
            try:
                # LLAMADA PARA 1024 DIMENSIONES (MODELO ESTABLE)
                embedding = genai.embed_content(
                    model="models/embedding-001",
                    content=prompt,
                    task_type="retrieval_query"
                )
                query_vector = embedding['embedding']
                
                # Búsqueda en Pinecone (Namespace correcto)
                results = index.query(
                    vector=query_vector, 
                    top_k=7, 
                    include_metadata=True,
                    namespace="silc-juridico"
                )
                
                contexto = "\n".join([res['metadata']['text'] for res in results['matches']])

                # Prompt Maestro
                full_prompt = f"""
                Eres el SILC, una IA experta en Derecho Mexicano de Rubio Intelligence Systems.
                Analiza la pregunta del usuario basándote en este contexto legal:
                
                {contexto}
                
                Pregunta: {prompt}
                """

                response = model.generate_content(full_prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                st.error(f"Error en el procesamiento: {e}")
