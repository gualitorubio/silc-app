import streamlit as st
from pinecone import Pinecone
import requests
import json

# 1. IDENTIDAD PROFESIONAL
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.markdown("---")

# 2. CONEXIÓN DIRECTA A RECURSOS
try:
    # Usamos Pinecone para los vectores (esto funciona correctamente)
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error(f"Fallo de inicio: {e}")

# 3. INTERFAZ DE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# 4. PROCESAMIENTO SIN LIBRERÍAS DE GOOGLE (BYPASS TOTAL)
if prompt := st.chat_input("Introduzca su consulta jurídica aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Vectorización vía Pinecone Inference (Evita errores de embedding de Google)
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            vector = res_embed[0].values

            # Búsqueda en Galaxia de Datos
            search = index.query(vector=vector, top_k=3, include_metadata=True, namespace="silc-juridico")
            contexto = "\n".join([r['metadata']['text'] for r in search['matches']])

            # PETICIÓN POST PURA (Forzamos v1 y el modelo flash)
            # Esta ruta es la más estable del mundo para Gemini
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"
            headers = {'Content-Type': 'application/json'}
            data = {
                "contents": [{"parts": [{"text": f"Eres el SILC. Contexto: {contexto}\nPregunta: {prompt}"}]}]
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error(f"El servidor respondió con código {response.status_code}. Detalle: {response.text}")
                
        except Exception as e:
            st.error(f"Error técnico en Rubio Intelligence Systems: {e}")

with st.sidebar:
    st.caption("© 2026 Rubio Intelligence Systems | Dr. Carlos Rubio")
