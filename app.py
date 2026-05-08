import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# 1. Configuración de Rubio Intelligence Systems
st.set_page_config(page_title="SILC - RIS", layout="wide")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")

# 2. Conexión de Infraestructura (Secrets)
try:
    # Google Gemini
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Pinecone (El verdadero motor de datos)
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos") # Su índice real
    
except Exception as e:
    st.error(f"Error de conexión: {e}")
    st.stop()

# 3. Lógica del Sistema
if prompt := st.chat_input("Consulta a la Galaxia de Datos..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Búsqueda Vectorial (Lo que hace al SILC único)
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            
            # Consultamos su base de leyes
            query_res = index.query(
                vector=res_embed[0].values, 
                top_k=5, 
                include_metadata=True,
                namespace="silc-juridico"
            )
            
            contexto = "\n\n".join([item['metadata']['text'] for item in query_res['matches']])
            
            # Generación con contexto real
            final_prompt = f"Contexto Legal:\n{contexto}\n\nPregunta: {prompt}"
            response = model.generate_content(final_prompt)
            
            st.markdown(response.text)
            
        except Exception as e:
            st.error(f"Error en Pinecone: {e}")
