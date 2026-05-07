import streamlit as st
from pinecone import Pinecone
import requests

# 1. IDENTIDAD PROFESIONAL
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")

with st.sidebar:
    st.header("Guía de Consulta")
    st.markdown("Analizando **Galaxia de Datos** (1024 dim).")
    st.divider()
    st.caption("© 2026 Rubio Intelligence Systems | Dr. Carlos Rubio")

st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Doctorando Carlos Rubio")

# 2. CONFIGURACIÓN DE RECURSOS
try:
    # Usamos Pinecone para todo el flujo de datos inicial
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error(f"Fallo de configuración inicial: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# 3. LÓGICA DE PROCESAMIENTO CON BYPASS DE RUTA
if prompt := st.chat_input("Introduzca su consulta jurídica aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Generación de vector usando el motor de Pinecone (Evita el 404 de Google)
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            vector = res_embed[0].values

            # Búsqueda semántica en los 87,508 registros
            search_results = index.query(vector=vector, top_k=5, include_metadata=True, namespace="silc-juridico")
            contexto = "\n".join([r['metadata']['text'] for r in search_results['matches']])

            # Bypass de API: Usamos la ruta más genérica posible para Gemini
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_KEY}"
            payload = {"contents": [{"parts": [{"text": f"Eres el SILC. Analiza este contexto legal mexicano: {contexto}\n\nPregunta: {prompt}"}]}]}
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error(f"El servidor de inteligencia no responde (Código {response.status_code}). Intente de nuevo en unos minutos.")
                
        except Exception as e:
            st.error(f"Error técnico en Rubio Intelligence Systems: {e}")
