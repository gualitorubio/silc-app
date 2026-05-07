import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os

# 1. CONFIGURACIÓN DE PÁGINA E IDENTIDAD
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️", layout="wide")

# Estilo personalizado para el título
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.subheader("Plataforma de Análisis Jurídico Avanzado")
st.info("Desarrollado por Rubio Intelligence Systems | Doctorando Carlos Rubio")

# 2. BARRA LATERAL (INSTRUCCIONES PARA EL USUARIO)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3439/3439002.png", width=100)
    st.header("Guía de Consulta")
    st.markdown("""
    **¿Cómo usar el SILC?**
    1. Escriba su duda jurídica en el chat de la derecha.
    2. El sistema consultará la base de datos **Galaxia de Datos** (87,000+ registros).
    3. La IA procesará la información y entregará un análisis técnico.
    
    **Ejemplos de consulta:**
    * *"¿Cuáles son las causas de utilidad pública en la Ley de Expropiación de 1936?"*
    * *"Diferencias entre el Código de Comercio actual y sus reformas recientes."*
    """)
    st.divider()
    st.caption("© 2026 Rubio Intelligence Systems | Versión Piloto Universitario")

# 3. CONFIGURACIÓN DE LLAVES Y MODELOS
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# 4. INICIALIZACIÓN DE RECURSOS (CACHÉ PARA VELOCIDAD)
@st.cache_resource
def init_resources():
    # Conexión a Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("galaxia-de-datos") 
    # Modelo de embeddings (debe coincidir con las dimensiones de tu index)
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return index, embed_model

try:
    index, embed_model = init_resources()
except Exception as e:
    st.error(f"Error crítico de conexión: {e}")

# 5. GESTIÓN DEL CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Mensaje de bienvenida del sistema
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Bienvenido al SILC. Estoy listo para analizar la legislación mexicana. ¿En qué área jurídica desea profundizar hoy?"
    })

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. LÓGICA DE PROCESAMIENTO (PREGUNTA -> PINECOE -> GEMINI)
if prompt := st.chat_input("Introduzca su consulta legal aquí..."):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando Rubio Intelligence Systems..."):
            try:
                # A. Crear vector de la pregunta
                query_vector = embed_model.encode(prompt).tolist()
                
                # B. Buscar en Pinecone (Namespace silc-juridico)
                results = index.query(
                    vector=query_vector, 
                    top_k=5, 
                    include_metadata=True,
                    namespace="silc-juridico"
                )
                
                # C. Extraer contexto
                contexto_legal = "\n".join([res['metadata']['text'] for res in results['matches']])

                # D. Construir Prompt Maestro
                full_prompt = f"""
                IDENTIDAD: Eres el SILC (Sistema de Inteligencia Legal y Contexto), una IA de élite desarrollada por Rubio Intelligence Systems.
                INSTRUCCIÓN: Responde de forma técnica, precisa y basada estrictamente en derecho mexicano.
                
                CONTEXTO LEGAL RECUPERADO:
                {contexto_legal}
                
                PREGUNTA DEL USUARIO:
                {prompt}
                
                RESPUESTA:
                """

                # E. Generar y mostrar respuesta
                response = model.generate_content(full_prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                st.error(f"Error en el procesamiento de Rubio Intelligence Systems: {e}")
