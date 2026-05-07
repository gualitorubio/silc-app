import streamlit as st
from pinecone import Pinecone
import requests
import json

# 1. IDENTIDAD PROFESIONAL
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")

with st.sidebar:
    st.header("Guía de Consulta")
    st.markdown("Analizando **Galaxia de Datos** (1024 dim).")
    st.divider()
    st.caption("© 2026 Rubio Intelligence Systems")

st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Doctorando Carlos Rubio")

# 2. CONFIGURACIÓN DE RECURSOS
try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error(f"Fallo de configuración: {e}")

# Función para bypass del error 404
def call_gemini_direct(prompt):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error de conexión directa (Status {response.status_code}): {response.text}"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# 3. PROCESAMIENTO
if prompt := st.chat_input("Consulta jurídica..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # USAMOS EL EMBEDDING DE PINECONE (Evita el 404 de Google Embeddings)
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            vector = res_embed[0].values

            # Búsqueda en la Galaxia de Datos
            search_results = index.query(vector=vector, top_k=5, include_metadata=True, namespace="silc-juridico")
            context = "\n".join([r['metadata']['text'] for r in search_results['matches']])

            # Respuesta vía Bypass Directo
            full_prompt = f"Eres el SILC de Rubio Intelligence Systems. Basado en este contexto: {context}\nPregunta: {prompt}"
            answer = call_gemini_direct(full_prompt)
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Error en Rubio Intelligence Systems: {e}")
