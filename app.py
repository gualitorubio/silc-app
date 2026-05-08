import streamlit as st
import requests
import json
from pinecone import Pinecone

# 1. Identidad
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal")
st.sidebar.write("Director: Doctorando Carlos Rubio")

# 2. Configuración de APIs (Secrets)
API_KEY = st.secrets["GEMINI_API_KEY"]
PINECONE_KEY = st.secrets["PINECONE_API_KEY"]

# Conexión a Base de Datos
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("galaxia-de-datos")

# 3. Interfaz de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Consulta jurídica..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # A. Recuperación de Contexto (Pinecone)
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            
            query_res = index.query(
                vector=res_embed[0].values, 
                top_k=3, 
                include_metadata=True,
                namespace="silc-juridico"
            )
            
            contexto = "\n".join([m['metadata']['text'] for m in query_res['matches']])
            
            # B. LLAMADA DIRECTA POR HTTP (Saltándose el SDK de Google)
            # Aquí obligamos a usar la versión 'v1' estable de pago
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": f"Eres el SILC. Contexto:\n{contexto}\n\nPregunta: {prompt}"}]
                }]
            }
            
            headers = {'Content-Type': 'application/json'}
            
            # Petición directa al servidor de Google v1
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            res_json = response.json()

            if response.status_code == 200:
                texto_final = res_json['candidates'][0]['content']['parts'][0]['text']
                st.markdown(texto_final)
                st.session_state.messages.append({"role": "assistant", "content": texto_final})
            else:
                # Si falla, nos dirá el error real de Google sin máscaras
                st.error(f"Error de Google: {res_json.get('error', {}).get('message', 'Error desconocido')}")

        except Exception as e:
            st.error(f"Aviso del Sistema: {str(e)}")
