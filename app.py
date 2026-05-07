import streamlit as st
from pinecone import Pinecone
import requests
import json

# 1. IDENTIDAD DEL SISTEMA
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")

with st.sidebar:
    st.header("Guía de Consulta")
    st.markdown("Analizando **Galaxia de Datos** (1024 dim).")
    st.divider()
    st.caption("© 2026 Rubio Intelligence Systems | Dr. Carlos Rubio")

st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Doctorando Carlos Rubio")

# 2. RECURSOS
try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error(f"Fallo de configuración: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# 3. PROCESAMIENTO CON BYPASS DE RUTA (VERSIÓN PRO)
if prompt := st.chat_input("Introduzca su consulta jurídica aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Vectorización vía Pinecone (para evitar el 404 de Google Embeddings)
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            vector = res_embed[0].values

            # Búsqueda en los 87,508 registros de leyes mexicanas
            search_results = index.query(vector=vector, top_k=5, include_metadata=True, namespace="silc-juridico")
            contexto = "\n".join([r['metadata']['text'] for r in search_results['matches']])

            # PETICIÓN MANUAL AL MODELO PRO (Ruta v1 estable)
            # Cambiamos a 'gemini-1.5-pro' para saltar el bloqueo de la versión flash
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent?key={API_KEY}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Eres el SILC de Rubio Intelligence Systems. Analiza profesionalmente:\nCONTEXTO:\n{contexto}\n\nPREGUNTA:\n{prompt}"
                    }]
                }],
                "generationConfig": {"temperature": 0.2}
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                # Si falla, mostramos el error técnico para diagnóstico
                st.error(f"Error de comunicación (Status {response.status_code}): {response.text}")
                
        except Exception as e:
            st.error(f"Error crítico en el motor de análisis: {e}")
