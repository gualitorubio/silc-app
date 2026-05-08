import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
import os

# 1. Identidad de Rubio Intelligence Systems
st.set_page_config(page_title="SILC - Inteligencia Legal", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")

# 2. Configuración de API de PAGO (Versión estable v1)
# Asegúrate de que en Secrets tengas GEMINI_API_KEY con la nueva llave
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Forzamos el uso del modelo de producción estable
model = genai.GenerativeModel('gemini-1.5-flash') 

# 3. Conexión a la Galaxia de Datos
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("galaxia-de-datos")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Motor de Respuesta
if prompt := st.chat_input("Consulta la Galaxia de Datos..."):
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
                top_k=5, 
                include_metadata=True,
                namespace="silc-juridico"
            )
            contexto = "\n\n".join([item['metadata']['text'] for item in query_res['matches']])
            
            # Generación con el modelo de pago
            instruccion = f"Eres el SILC (Rubio Intelligence Systems). Analiza con rigor jurídico.\nContexto: {contexto}\nPregunta: {prompt}"
            
            # Cambiamos la forma de llamar a la respuesta para evitar el error 404
            response = model.generate_content(instruccion)
            
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"Aviso Técnico: {str(e)}")
