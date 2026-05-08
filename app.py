import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# 1. Configuración Básica
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal")
st.sidebar.write("Director: Doctorando Carlos Rubio")

# 2. Inicialización de APIs
# Usamos la llave de pago para evitar límites
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("galaxia-de-datos")

# 3. Historial del Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Procesamiento de Consultas
if prompt := st.chat_input("Escriba su consulta jurídica..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # A. Recuperación de Contexto (Pinecone)
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
        
        contexto = ""
        for match in query_res['matches']:
            contexto += match['metadata']['text'] + "\n\n"
        
        # B. Generación de Respuesta (Gemini)
        # El uso de f-strings aquí es lo más estable para la API v1
        prompt_final = f"Eres el SILC. Usa este contexto:\n{contexto}\n\nPregunta: {prompt}"
        
        try:
            # Esta llamada usa tu bono de pago automáticamente
            response = model.generate_content(prompt_final)
            texto_respuesta = response.text
            st.markdown(texto_respuesta)
            st.session_state.messages.append({"role": "assistant", "content": texto_respuesta})
        except Exception as e:
            st.error(f"Error en Gemini: {str(e)}")
