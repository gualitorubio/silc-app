import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# IDENTIDAD
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Dr. Carlos Rubio")

# CONFIGURACIÓN DE LLAVES
try:
    # 1. Configurar Gemini con la librería oficial
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # 2. Configurar Pinecone
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
except Exception as e:
    st.error(f"Fallo en la carga de credenciales: {e}")

# MOTOR DE INTELIGENCIA CON FALLBACK
def generar_respuesta_segura(prompt_final):
    # Intentamos primero con Flash (más rápido), si falla, vamos a Pro
    for model_name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt_final)
            return response.text
        except Exception:
            continue
    return "Error Crítico: Ningún modelo de Google respondió. Verifique que la API esté habilitada en su consola de Google Cloud."

# INTERFAZ
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
            
            # Generación
            full_prompt = f"Eres el SILC. Analiza este contexto legal:\n{contexto}\n\nPregunta: {prompt}"
            respuesta = generar_respuesta_segura(full_prompt)
            st.markdown(respuesta)
            
        except Exception as e:
            st.error(f"Error técnico en el motor SILC: {e}")
