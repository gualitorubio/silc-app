import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# 1. IDENTIDAD JURÍDICA
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Dr. Carlos Rubio")

# 2. CONFIGURACIÓN DE RECURSOS
try:
    # Configuración de IA con la llave que generó a las 4:37 p.m.
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Configuración de la Galaxia de Datos
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
except Exception as e:
    st.error(f"Fallo en credenciales: {e}")

# 3. INTERFAZ DE CONSULTA
if prompt := st.chat_input("Introduzca su consulta jurídica aquí..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Búsqueda semántica en los 1024 dim
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            
            search = index.query(
                vector=res_embed[0].values, 
                top_k=4, 
                include_metadata=True, 
                namespace="silc-juridico"
            )
            contexto = "\n".join([r['metadata']['text'] for r in search['matches']])

            # GENERACIÓN CON FALLBACK AUTOMÁTICO
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                f"Eres el SILC. Analiza este contexto legal:\n{contexto}\n\nPregunta: {prompt}"
            )
            
            st.markdown(response.text)
            
        except Exception as e:
            st.error(f"Error técnico en Rubio Intelligence Systems: {e}")
            st.warning("Verifique que 'google-generativeai' esté en su requirements.txt")
