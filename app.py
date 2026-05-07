import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# 1. IDENTIDAD DEL SISTEMA
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Dr. Carlos Rubio")

# 2. CARGA DE SEGURIDAD
try:
    # Usamos la configuración global
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
except Exception as e:
    st.error(f"Error de configuración: {e}")

# 3. INTERFAZ Y LÓGICA JURÍDICA
if prompt := st.chat_input("Introduzca su consulta jurídica aquí..."):
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            # Recuperación desde la Galaxia de Datos
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            search = index.query(vector=res_embed[0].values, top_k=4, include_metadata=True, namespace="silc-juridico")
            contexto = "\n".join([r['metadata']['text'] for r in search['matches']])

            # MOTOR DE GENERACIÓN: Forzamos la versión 1.0 Pro
            # Este modelo es el que tiene la mayor tasa de éxito cuando la API es nueva
            model = genai.GenerativeModel('gemini-pro')
            
            intentos = 0
            respuesta_obtenida = False
            
            while intentos < 2 and not respuesta_obtenida:
                try:
                    response = model.generate_content(
                        f"Eres el SILC de Rubio Intelligence Systems. Analiza este contexto:\n{contexto}\n\nPregunta: {prompt}"
                    )
                    st.markdown(response.text)
                    respuesta_obtenida = True
                except Exception:
                    intentos += 1
                    # Si falla el Pro, intentamos con Flash como último recurso
                    model = genai.GenerativeModel('gemini-1.5-flash')
            
            if not respuesta_obtenida:
                st.error("El servidor de Google aún no autoriza su conexión. Por favor, genere una NUEVA clave en AI Studio e inténtelo de nuevo.")

        except Exception as e:
            st.error(f"Fallo técnico: {e}")
