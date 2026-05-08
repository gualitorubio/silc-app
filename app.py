import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# ==========================================
# 1. CONFIGURACIÓN DE IDENTIDAD DEL SILC
# ==========================================
st.set_page_config(
    page_title="SILC - Rubio Intelligence Systems", 
    page_icon="⚖️", 
    layout="centered"
)

st.title("⚖️ SILC: Sistema de Inteligencia Legal")
st.subheader("Plataforma de Inteligencia Jurídica y Convencional")

# Barra lateral con identidad profesional
st.sidebar.markdown("### Rubio Intelligence Systems")
st.sidebar.write("**Director:** Doctorando Carlos Rubio")
st.sidebar.markdown("---")

# ==========================================
# 2. CONEXIÓN DE MOTOR (GEMINI 3 & PINECONE)
# ==========================================

# Usamos la nueva API Key de AI Studio y el modelo Gemini 3
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Cambiamos a gemini-3-flash-preview para evitar el error 404 del modelo 1.5
    model = genai.GenerativeModel('gemini-3-flash-preview')
except Exception as e:
    st.error(f"Error en configuración de IA: {e}")

# Conexión a la Galaxia de Datos
try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
except Exception as e:
    st.error(f"Error en base de datos Pinecone: {e}")

# ==========================================
# 3. GESTIÓN DEL CHAT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 4. PROCESAMIENTO JURÍDICO (RAG)
# ==========================================
if prompt := st.chat_input("Escriba su consulta jurídica aquí..."):
    # Guardar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # A. Recuperación de contexto legal
            with st.spinner("Analizando legislación y precedentes..."):
                res_embed = pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=[prompt],
                    parameters={"input_type": "query"}
                )
                
                query_res = index.query(
                    vector=res_embed[0].values, 
                    top_k=5, 
                    include_metadata=True,
                    namespace="silc-juridico"
                )
                
                contexto_legal = "\n\n".join([m['metadata']['text'] for m in query_res['matches']])

            # B. Generación de respuesta con Gemini 3
            instruccion_maestra = (
                f"Eres el SILC (Sistema de Inteligencia Legal y Contexto). "
                f"Analiza la consulta con rigor jurídico basándote en este contexto:\n\n"
                f"{contexto_legal}\n\n"
                f"PREGUNTA:\n{prompt}"
            )

            # Esta llamada usará tu nueva cuota de AI Studio
            response = model.generate_content(instruccion_maestra)
            
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

        except Exception as e:
            # Diagnóstico en caso de error
            st.error(f"Aviso del Sistema: {str(e)}")
            if "404" in str(e):
                st.info("Sugerencia: Dale a 'Reboot App' en el menú de la derecha de Streamlit.")

# Pie de página
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Rubio Intelligence Systems.")
