import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
import os

# 1. IDENTIDAD VISUAL
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.sidebar.markdown("### Rubio Intelligence Systems")
st.sidebar.write("Director: Doctorando Carlos Rubio")

# 2. CONFIGURACIÓN DE APIS (USANDO TUS SECRETS)
# La API de pago requiere la librería actualizada para evitar el error 404
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("galaxia-de-datos")

# 3. GESTIÓN DE MEMORIA
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. MOTOR DE RESPUESTA JURÍDICA
if prompt := st.chat_input("Escriba su consulta jurídica..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Búsqueda en Pinecone
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
            
            contexto = "\n\n".join([item['metadata']['text'] for item in query_res['matches']])
            
            # Generación de Respuesta con IA de Pago
            system_prompt = (
                f"Eres el SILC, una IA experta en derecho mexicano y convencionalidad. "
                f"Analiza la siguiente pregunta basándote en este contexto legal:\n\n{contexto}"
            )
            
            # Llamada directa al modelo estable
            response = model.generate_content([system_prompt, prompt])
            
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"Aviso del Sistema: {str(e)}")
