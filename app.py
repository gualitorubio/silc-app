import streamlit as st
from pinecone import Pinecone
import requests

# 1. IDENTIDAD JURÍDICA
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Dr. Carlos Rubio")

# 2. CONEXIÓN DE RECURSOS
try:
    # Verificamos la existencia de las llaves en Secrets
    if "GEMINI_API_KEY" not in st.secrets or "PINECONE_API_KEY" not in st.secrets:
        st.error("Error: Faltan las llaves en la configuración de Secrets.")
    
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error(f"Fallo de inicialización: {e}")

# 3. INTERFAZ DE CHAT
if prompt := st.chat_input("Introduzca su consulta jurídica aquí..."):
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Recuperación de contexto desde la Galaxia de Datos (Pinecone)
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            
            search = index.query(vector=res_embed[0].values, top_k=4, include_metadata=True, namespace="silc-juridico")
            contexto = "\n".join([r['metadata']['text'] for r in search['matches']])

            # BYPASS DE ERROR 404: Usamos la ruta v1 con el modelo flash
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"
            
            payload = {
                "contents": [{"parts": [{"text": f"Eres el SILC. Analiza este contexto legal:\n{contexto}\n\nPregunta: {prompt}"}]}]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                st.markdown(answer)
            else:
                # Si falla, mostramos el mensaje de error de Google para diagnóstico final
                st.error(f"Error de servidor (Status {response.status_code})")
                st.write("Respuesta técnica de Google:", response.text)
                
        except Exception as e:
            st.error(f"Fallo en Rubio Intelligence Systems: {e}")
