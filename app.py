import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
import os

# ==========================================
# 1. CONFIGURACIÓN DE IDENTIDAD Y PANTALLA
# ==========================================
st.set_page_config(
    page_title="SILC - Rubio Intelligence Systems", 
    page_icon="⚖️", 
    layout="centered"
)

# Estética Profesional
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stTitle { color: #1e3a8a; font-family: 'Helvetica', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.subheader("Plataforma de Inteligencia Jurídica y Convencional")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100) # Icono genérico de justicia
st.sidebar.markdown("---")
st.sidebar.write("**Director:** Doctorando Carlos Rubio")
st.sidebar.write("**Powered by:** Rubio Intelligence Systems")

# ==========================================
# 2. CONEXIÓN DE PAGO (GOOGLE & PINECONE)
# ==========================================

# Configuración de Google Gemini (Versión de Pago v1)
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Forzamos la versión v1 para evitar el error 404 de la v1beta
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash'
    )
except Exception as e:
    st.error(f"Error de configuración en Google AI: {e}")

# Configuración de Pinecone (Galaxia de Datos)
try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
except Exception as e:
    st.error(f"Error de conexión con la base de datos legal: {e}")

# ==========================================
# 3. GESTIÓN DEL HISTORIAL DE CHAT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 4. MOTOR DE INFERENCIA JURÍDICA (RAG)
# ==========================================
if prompt := st.chat_input("Escriba su consulta jurídica aquí..."):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # A. Búsqueda de Contexto en la Galaxia de Datos
            with st.spinner("Consultando legislación y jurisprudencia..."):
                res_embed = pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=[prompt],
                    parameters={"input_type": "query"}
                )
                
                query_res = index.query(
                    vector=res_embed[0].values, 
                    top_k=5, 
                    include_metadata=True,
                    namespace="silc-juridico" # Asegúrate de que este sea tu namespace
                )
                
                contexto_legal = "\n\n".join([item['metadata']['text'] for item in query_res['matches']])

            # B. Generación de Respuesta Especializada
            instruccion_maestra = (
                f"Eres el SILC (Sistema de Inteligencia Legal y Contexto), una IA experta en derecho mexicano "
                f"y control de convencionalidad. Utiliza el siguiente contexto legal para responder con precisión. "
                f"Si el contexto no contiene la respuesta, utiliza tu base de conocimiento pero acláralo.\n\n"
                f"CONTEXTO LEGAL RECUPERADO:\n{contexto_legal}\n\n"
                f"PREGUNTA DEL JURISTA:\n{prompt}"
            )

            # Llamada al modelo (Usa automáticamente la cuota de tu bono de $300 USD)
            response = model.generate_content(instruccion_maestra)
            
            # Mostrar respuesta y guardar en historial
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

        except Exception as e:
            # Manejo de errores detallado para depuración
            if "404
