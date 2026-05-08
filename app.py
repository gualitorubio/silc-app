import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# 1. Interfaz RIS
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", layout="wide")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")

st.sidebar.title("Infraestructura RIS")
st.sidebar.info("v17.4.12.25\nDirector: Dr. Carlos Rubio\nEstado: Piloto Universitario")

# 2. Conexión limpia (Sin v1beta)
try:
    # Configuración base
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # IMPORTANTE: Usamos solo 'gemini-1.5-flash' sin sufijos ni betas
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    
except Exception as e:
    st.error(f"Error de infraestructura: {e}")
    st.stop()

# 3. Procesamiento Jurídico
if prompt := st.chat_input("Introduzca su consulta jurídica..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Búsqueda en Pinecone
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
            
            contexto = "\n\n".join([item['metadata']['text'] for item in query_res['matches']])
            
            # Generación de respuesta
            response = model.generate_content(f"Contexto: {contexto}\n\nPregunta: {prompt}")
            st.markdown(response.text)
            
        except Exception as e:
            st.error(f"Error en el procesamiento: {e}")

st.markdown("---")
st.caption("Powered by Rubio Intelligence Systems © 2026")
