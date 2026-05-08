import streamlit as st
import google.generativeai as genai
from google.generativeai.types import RequestOptions
from pinecone import Pinecone

# 1. Configuración de Identidad
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal")
st.sidebar.markdown("### Rubio Intelligence Systems")
st.sidebar.write("Director: Doctorando Carlos Rubio")

# 2. Conexión de Pago (FUERZA VERSIÓN v1)
try:
    # Configuramos la llave
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Esta es la parte CRÍTICA: forzamos explícitamente la versión 'v1' 
    # para que Google no te mande a la 'v1beta' que da el error 404.
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash'
    )
    # Definimos opciones de envío para asegurar la ruta de pago
    envio_pro = RequestOptions(api_version='v1')
    
except Exception as e:
    st.error(f"Error de configuración de IA: {e}")

# 3. Conexión a Base de Datos (Pinecone)
try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
except Exception as e:
    st.error(f"Error de base de datos: {e}")

# 4. Interfaz de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escriba su consulta jurídica..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Búsqueda Vectorial
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            
            query_res = index.query(
                vector=res_embed[0].values, 
                top_k=3, 
                include_metadata=True,
                namespace="silc-juridico"
            )
            
            contexto = "\n\n".join
