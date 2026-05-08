import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# 1. Configuración de Rubio Intelligence Systems
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", layout="wide")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.sidebar.info("v17.4.12.25 | Director: Dr. Carlos Rubio")

# 2. Conexión de Infraestructura (Secrets)
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Conexión real a su Galaxia de Datos
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    
except Exception as e:
    st.error(f"Error de infraestructura: {e}")
    st.stop()

# 3. Procesamiento Jurídico RAG (Verdadero SILC)
if prompt := st.chat_input("Realice su consulta sobre legislación mexicana..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando Galaxia de Datos..."):
            try:
                # Búsqueda semántica en Pinecone
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
                
                contexto_legal = "\n\n".join([item['metadata']['text'] for item in query_res['matches']])
                
                # Generación de respuesta con base en sus 317 leyes
                prompt_final = f"Usa este contexto legal:\n{contexto_legal}\n\nPregunta: {prompt}"
                response = model.generate_content(prompt_final)
                
                st.markdown(response.text)
                
                with st.expander("Fuentes detectadas en Pinecone"):
                    for match in query_res['matches']:
                        st.write(f"📖 {match['metadata'].get('source', 'Ley')}")

            except Exception as e:
                st.error(f"Error en el procesamiento: {e}")
