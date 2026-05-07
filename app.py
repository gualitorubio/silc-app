import streamlit as st
from pinecone import Pinecone
import requests

# 1. IDENTIDAD PROFESIONAL
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Dr. Carlos Rubio")

# 2. CONEXIÓN DE RECURSOS (Desde Secrets)
try:
    # Conexión a la Galaxia de Datos
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    
    # Nueva llave de Google AI Studio
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error(f"Fallo de configuración: {e}")

# 3. INTERFAZ DE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# 4. LÓGICA JURÍDICA (RAG)
if prompt := st.chat_input("Introduzca su consulta jurídica aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # A. Vectorización (Pinecone Inference)
            # Usamos el modelo e5 para las 1024 dimensiones de tu índice
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            vector = res_embed[0].values

            # B. Búsqueda Semántica
            search = index.query(
                vector=vector, 
                top_k=5, 
                include_metadata=True, 
                namespace="silc-juridico"
            )
            contexto = "\n".join([r['metadata']['text'] for r in search['matches']])

            # C. Generación de Respuesta (Bypass de Error 404)
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
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error(f"Error de conexión (Status {response.status_code}). Revisa si la clave en Secrets es la correcta.")
                
        except Exception as e:
            st.error(f"Error técnico en Rubio Intelligence Systems: {e}")
