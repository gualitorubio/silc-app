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
    st.markdown("Consultando la **Galaxia de Datos** (1024 dim).")
    st.divider()
    st.caption("© 2026 Rubio Intelligence Systems")

# 3. CONFIGURACIÓN DE LLAVES
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

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bienvenido al SILC. El sistema está en línea. ¿Qué análisis legal desea realizar?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. PROCESAMIENTO
if prompt := st.chat_input("Introduzca su consulta legal aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando en la Galaxia de Datos..."):
            try:
                # LLAMADA COMPATIBLE PARA 1024 DIMENSIONES
                embedding = genai.embed_content(
                    model="models/embedding-001",
                    content=prompt,
                    task_type="retrieval_query"
                )
                query_vector = embedding['embedding']
                
                # Búsqueda en Pinecone
                results = index.query(
                    vector=query_vector, 
                    top_k=5, 
                    include_metadata=True,
                    namespace="silc-juridico"
                )
                
                contexto = "\n".join([res['metadata']['text'] for res in results['matches']])

                full_prompt = f"Eres el SILC de Rubio Intelligence Systems. Usa este contexto: {contexto}. Pregunta: {prompt}"

                response = model.generate_content(full_prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                st.error(f"Error: {e}")    return index

index = init_resources()

# 5. GESTIÓN DEL CHAT
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bienvenido al SILC. La conexión con la Galaxia de Datos es estable. ¿Qué análisis legal desea realizar?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. LÓGICA DE PROCESAMIENTO
if prompt := st.chat_input("Introduzca su consulta legal aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Procesando en Rubio Intelligence Systems (Dimension: 1024)..."):
            try:
                # CAMBIO CLAVE: Usamos el modelo de Google que genera 1024 dimensiones
                embedding_result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=prompt,
                    task_type="retrieval_query"
                )
                query_vector = embedding_result['embedding']
                
                # Búsqueda en Pinecone
                results = index.query(
                    vector=query_vector, 
                    top_k=5, 
                    include_metadata=True,
                    namespace="silc-juridico"
                )
                
                contexto_legal = "\n".join([res['metadata']['text'] for res in results['matches']])

                full_prompt = f"""
                IDENTIDAD: Eres el SILC, una IA de Rubio Intelligence Systems.
                INSTRUCCIÓN: Responde de forma técnica y profunda.
                CONTEXTO LEGAL: {contexto_legal}
                PREGUNTA: {prompt}
                """

                response = model.generate_content(full_prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                st.error(f"Error de dimensión o procesamiento: {e}")
