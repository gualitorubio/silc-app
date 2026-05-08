import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# 1. Identidad RIS
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", layout="wide")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")

# 2. Conexión Estable
try:
    # Google - Usamos la versión estable para evitar el 404 de la v1beta
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Pinecone - Conexión a la Galaxia de Datos
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
except Exception as e:
    st.error(f"Falla de Infraestructura: {e}")
    st.stop()

# 3. Consulta RAG (El verdadero SILC)
if prompt := st.chat_input("Consulta a la Galaxia de Datos..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Búsqueda de vectores
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            
            query_res = index.query(
                vector=res_embed[0].values, 
                top_k=5, 
                include_metadata=True,
                namespace="silc-juridico"
            )
            
            contexto = "\n\n".join([item['metadata']['text'] for item in query_res['matches']])
            
            # Respuesta Jurídica
            response = model.generate_content(f"Contexto:\n{contexto}\n\nPregunta: {prompt}")
            st.markdown(response.text)
            
        except Exception as e:
            st.error(f"Error en el motor de datos: {e}")
