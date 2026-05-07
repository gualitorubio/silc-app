import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# 1. IDENTIDAD E INTERFAZ (CARGA INMEDIATA)
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️", layout="wide")

# Barra lateral recuperada
with st.sidebar:
    st.header("Guía de Consulta")
    st.markdown("""
    **Instrucciones:**
    1. Ingrese su consulta jurídica abajo.
    2. El sistema buscará en la **Galaxia de Datos** (1024 dim).
    3. Recibirá un análisis técnico basado en legislación mexicana.
    """)
    st.divider()
    st.caption("© 2026 Rubio Intelligence Systems | Doctorando Carlos Rubio")

st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Doctorando Carlos Rubio")

# 2. CONFIGURACIÓN DE RECURSOS CON CONTROL DE ERRORES
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    
    # Configuración forzada a versión estable v1
    genai.configure(api_key=GEMINI_API_KEY, transport='rest') 
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("galaxia-de-datos")
except Exception as e:
    st.error(f"Fallo en la inicialización: {e}")

# 3. GESTIÓN DEL CHAT
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bienvenido al SILC. Sistema conectado a la Galaxia de Datos. ¿Qué área del derecho analizaremos?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. LÓGICA DE PROCESAMIENTO (REPARACIÓN DEL 404)
if prompt := st.chat_input("Introduzca su consulta legal aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Procesando en Rubio Intelligence Systems..."):
            try:
                # LLAMADA MANUAL AL EMBEDDING PARA EVITAR ERROR DE VERSIÓN
                embedding_data = genai.embed_content(
                    model="models/embedding-001",
                    content=prompt,
                    task_type="retrieval_query"
                )
                query_vector = embedding_data['embedding']
                
                # Búsqueda en Pinecone
                results = index.query(
                    vector=query_vector, 
                    top_k=5, 
                    include_metadata=True,
                    namespace="silc-juridico"
                )
                
                contexto = "\n".join([res['metadata']['text'] for res in results['matches']])

                full_prompt = f"""
                Eres el SILC de Rubio Intelligence Systems. 
                Usa este contexto legal para responder:
                {contexto}
                
                Pregunta: {prompt}
                """

                response = model.generate_content(full_prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                st.error(f"Error de conexión con el motor de análisis: {e}")
