import streamlit as st
from pinecone import Pinecone
import requests

# IDENTIDAD PROFESIONAL
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")

st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Dr. Carlos Rubio")

# CONFIGURACIÓN
try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error(f"Error de configuración: {e}")

# PROCESAMIENTO
if prompt := st.chat_input("Consulta jurídica..."):
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. Vectorización (Pinecone nativo)
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            vector = res_embed[0].values

            # 2. Búsqueda
            results = index.query(vector=vector, top_k=5, include_metadata=True, namespace="silc-juridico")
            contexto = "\n".join([r['metadata']['text'] for r in results['matches']])

            # 3. LLAMADA DIRECTA (Sin librerías intermedias)
            # Forzamos gemini-1.5-flash que es la versión real y estable
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
            payload = {"contents": [{"parts": [{"text": f"Contexto: {contexto}\nPregunta: {prompt}"}]}]}
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                st.markdown(response.json()['candidates'][0]['content']['parts'][0]['text'])
            else:
                st.error(f"Error 404: El modelo especificado no existe o la ruta es inválida (Status {response.status_code})")
        except Exception as e:
            st.error(f"Error técnico: {e}")
