import streamlit as st
from pinecone import Pinecone
import requests

# 1. IDENTIDAD DEL SISTEMA
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Dr. Carlos Rubio")

# 2. CONFIGURACIÓN (Secrets)
try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error(f"Error de llaves: {e}")

# 3. CHAT E INTELIGENCIA
if prompt := st.chat_input("Introduzca su consulta jurídica..."):
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Búsqueda en Galaxia de Datos
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            
            search = index.query(vector=res_embed[0].values, top_k=4, include_metadata=True, namespace="silc-juridico")
            contexto = "\n".join([r['metadata']['text'] for r in search['matches']])

            # LA LLAMADA DEFINITIVA: Usamos 'gemini-pro' (La ruta más estable del mundo)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEY}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Actúa como el SILC de Rubio Intelligence Systems. Analiza este contexto legal:\n{contexto}\n\nPregunta: {prompt}"
                    }]
                }]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                st.markdown(answer)
            else:
                # Si esto falla, el problema es la propagación de la llave
                st.error(f"El servidor de inteligencia aún no reconoce la nueva llave (Código {response.status_code}).")
                st.write("Respuesta de Google:", response.text)
                
        except Exception as e:
            st.error(f"Fallo en Rubio Intelligence Systems: {e}")
