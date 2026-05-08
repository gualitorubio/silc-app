import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# ==========================================
# 1. CONFIGURACIÓN DE IDENTIDAD Y UI
# ==========================================
st.set_page_config(
    page_title="SILC - Rubio Intelligence Systems", 
    page_icon="⚖️", 
    layout="wide"
)

# Estilo profesional para Rubio Intelligence Systems
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.markdown("---")

st.sidebar.title("Infraestructura RIS")
st.sidebar.info("""
**Versión:** 17.4.12.25  
**Director:** Dr. Carlos Rubio  
**Estado:** Piloto Universitario
""")

# ==========================================
# 2. CONEXIÓN DE INFRAESTRUCTURA
# ==========================================
try:
    # Configuración de Google Gemini (Versión estable)
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Se utiliza la versión más reciente para evitar Error 404
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # Configuración de Pinecone para la Galaxia de Datos
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    
except Exception as e:
    st.error(f"❌ Error de conexión: {e}")
    st.info("Verifique que las llaves en 'Secrets' no tengan espacios y que el índice de Pinecone esté activo.")
    st.stop()

# ==========================================
# 3. INTERFAZ DE CONSULTA JURÍDICA
# ==========================================
if prompt := st.chat_input("Introduzca su consulta sobre la Reforma de 2011 o la Ley de 1936..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando convencionalidad en la Galaxia de Datos..."):
            try:
                # A. Generar Embedding (Búsqueda semántica)
                res_embed = pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=[prompt],
                    parameters={"input_type": "query"}
                )
                
                # B. Recuperación de Contexto Legal
                query_res = index.query(
                    vector=res_embed[0].values, 
                    top_k=5, 
                    include_metadata=True, 
                    namespace="silc-juridico"
                )
                
                contexto_recuperado = "\n\n".join([item['metadata']['text'] for item in query_res['matches']])
                
                # C. Generación de Respuesta Especializada
                instruccion_silc = f"""
                Actúa como el SILC (Sistema de Inteligencia Legal y Contexto). 
                Analiza la consulta del Doctorando Carlos Rubio basándote en:
                1. El contexto legal recuperado de la base de datos.
                2. El principio de convencionalidad y la Reforma Constitucional de 2011.
                
                CONTEXTO:
                {contexto_recuperado}
                
                CONSULTA:
                {prompt}
                """
                
                respuesta = model.generate_content(instruccion_silc)
                st.markdown(respuesta.text)
                
                # D. Transparencia de Fuentes
                with st.expander("Fuentes consultadas (317 Leyes Federales)"):
                    for match in query_res['matches']:
                        st.write(f"📌 {match['metadata'].get('source', 'Documento Legal')}")
                        st.caption(match['metadata']['text'][:250] + "...")

            except Exception as e:
                st.error(f"Error en el procesamiento: {e}")

# ==========================================
# 4. PIE DE PÁGINA
# ==========================================
st.markdown("---")
st.caption("Powered by Rubio Intelligence Systems © 2026 | Análisis de Proporcionalidad y Debido Proceso")
