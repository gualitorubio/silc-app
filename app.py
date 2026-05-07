import streamlit as st
from pinecone import Pinecone
import requests

# IDENTIDAD PROFESIONAL
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Dr. Carlos Rubio")

# CONFIGURACIÓN DE RECURSOS
try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error(f"Fallo de configuración: {e}")

# LÓGICA DE CONSULTA
if prompt := st.chat_input("Introduzca su consulta jurídica..."):
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. Vectorización Nativa de Pinecone (Evitamos Google aquí para evitar 404)
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            vector = res_embed[0].values

            # 2. Búsqueda en Galaxia de Datos (87,508 registros)
            results = index.query(vector=vector, top_k=5, include_metadata=True, namespace="silc-juridico")
            contexto = "\n".join([r['metadata']['text'] for r in results['matches']])

            # 3. LLAMADA DIRECTA A GEMINI 1.5-FLASH (La versión correcta)
            # Usamos v1beta porque es la que acepta peticiones REST directas con mayor estabilidad
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Eres el SILC de Rubio Intelligence Systems. Basado en este contexto legal mexicano:\n{contexto}\n\nPregunta: {prompt}"
                    }]
                }]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                respuesta = response.json()['candidates'][0]['content']['parts'][0]['text']
                st.markdown(respuesta)
            else:
                st.error(f"Error de API (Status {response.status_code}): {response.text}")
                
        except Exception as e:
            st.error(f"Error en Rubio Intelligence Systems: {e}")
