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

st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.markdown("---")
st.sidebar.image("https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_92x30dp.png", width=100) # Placeholder para logo RIS
st.sidebar.title("Configuración")
st.sidebar.info("v17.4.12.25 | Rubio Intelligence Systems\n\nDirector: Dr. Carlos Rubio")

# ==========================================
# 2. CONEXIÓN DE INFRAESTRUCTURA (SECRETS)
# ==========================================
try:
    # Configuración de Google Gemini
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Configuración de Pinecone
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
    
    # Modelo optimizado para evitar Error 404
    model = genai.GenerativeModel('gemini-1.5-flash')
    
except Exception as e:
    st.error(f"❌ Error en la carga de infraestructura: {e}")
    st.stop()

# ==========================================
# 3. LÓGICA DE PROCESAMIENTO JURÍDICO
# ==========================================
if prompt := st.chat_input("Realice su consulta sobre legislación mexicana (ej. Reforma 2011)..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando Galaxia de Datos y analizando convencionalidad..."):
            try:
                # A. Generar Embedding de la consulta
                res_embed = pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=[prompt],
                    parameters={"input_type": "query"}
                )
                
                # B. Búsqueda Vectorial en Pinecone
                # Buscamos en el namespace 'silc-juridico' donde están las 317 leyes
                query_res = index.query(
                    vector=res_embed[0].values, 
                    top_k=5, 
                    include_metadata=True, 
                    namespace="silc-juridico"
                )
                
                # C. Construcción del Contexto
                contexto_legal = "\n\n".join([item['metadata']['text'] for item in query_res['matches']])
                
                # D. Generación de Respuesta con IA
                prompt_final = f"""
                Eres el SILC (Sistema de Inteligencia Legal y Contexto) de Rubio Intelligence Systems.
                Analiza la siguiente consulta basándote en el contexto legal proporcionado y los principios 
                de la Reforma Constitucional de 2011 (Derechos Humanos y Pro Personae).

                CONTEXTO LEGAL RECUPERADO:
                {contexto_legal}

                CONSULTA DEL USUARIO:
                {prompt}

                RESPUESTA JURÍDICA:
                """
                
                response = model.generate_content(prompt_final)
                
                # E. Despliegue de Resultados
                st.markdown(response.text)
                
                with st.expander("Ver fuentes de la Galaxia de Datos"):
                    for match in query_res['matches']:
                        st.write(f"**Fuente:** {match['metadata'].get('source', 'Ley Federal')}")
                        st.caption(match['metadata']['text'][:300] + "...")

            except Exception as e:
                st.error(f"Hubo un error en el procesamiento: {e}")
                st.info("Sugerencia: Verifique que el índice 'galaxia-de-datos' esté activo en su consola de Pinecone.")

# ==========================================
# 4. PIE DE PÁGINA PROFESIONAL
# ==========================================
st.markdown("---")
st.caption("Powered by Rubio Intelligence Systems © 2026 | Análisis de Convencionalidad y Celeridad Procesal")
