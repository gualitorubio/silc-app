import streamlit as st
from pinecone import Pinecone
import requests

# 1. IDENTIDAD
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Dr. Carlos Rubio")

# 2. CONFIGURACIÓN
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

# 3. LÓGICA DE PROCESAMIENTO
if prompt := st.chat_input("Introduzca su consulta jurídica aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Vectorización vía Pinecone Inference (Para evitar el 404 de embeddings de Google)
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            vector = res_embed[0].values

            # Búsqueda en Galaxia de Datos
            search_results = index.query(vector=vector, top_k=3, include_metadata=True, namespace="silc-juridico")
            contexto = "\n".join([r['metadata']['text'] for r in search_results['matches']])

            # LA LLAMADA DE RESCATE: Usamos 'gemini-pro' (sin 1.5)
            # Esta es la ruta más compatible y antigua de la API v1
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={API_KEY}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Eres el SILC de Rubio Intelligence Systems. Contexto legal:\n{contexto}\n\nPregunta: {prompt}"
                    }]
                }]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                # Si falla, mostramos el error exacto para diagnosticar la API Key
                st.error(f"Respuesta del servidor: {response.status_code}. Verifique si su API Key tiene permisos para Gemini Pro.")
                
        except Exception as e:
            st.error(f"Error técnico: {e}")
