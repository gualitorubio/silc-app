import streamlit as st
from pinecone import Pinecone
import requests

# 1. IDENTIDAD JURÍDICA
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Dr. Carlos Rubio")

# 2. RECURSOS DESDE SECRETS
try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error(f"Fallo de configuración: {e}")

# 3. INTERFAZ
if prompt := st.chat_input("Introduzca su consulta jurídica aquí..."):
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Recuperación de la Galaxia de Datos (1024 dim)
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            
            search = index.query(vector=res_embed[0].values, top_k=4, include_metadata=True, namespace="silc-juridico")
            contexto = "\n".join([r['metadata']['text'] for r in search['matches']])

            # BYPASS FINAL: Usamos la ruta v1beta para forzar la detección del modelo
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Eres el SILC de Rubio Intelligence Systems. Analiza este contexto legal:\n{contexto}\n\nPregunta: {prompt}"
                    }]
                }]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                st.markdown(answer)
            else:
                st.error(f"Error de servidor: {response.status_code}")
                st.write("Respuesta técnica:", response.text)
                
        except Exception as e:
            st.error(f"Fallo técnico: {e}")
