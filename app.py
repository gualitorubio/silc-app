import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# 1. Configuración de Identidad y Estética
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal")
st.sidebar.markdown("### Rubio Intelligence Systems")
st.sidebar.write("Director: Doctorando Carlos Rubio")

# 2. Conexión de Pago (Forzando versión estable v1)
# Al usar genai.GenerativeModel sin especificar versión, la librería 
# detecta tu API Key de pago y usa la ruta de producción.
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Error de configuración de IA: {e}")

# 3. Conexión a Base de Datos (Pinecone)
try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
except Exception as e:
    st.error(f"Error de base de datos: {e}")

# 4. Interfaz de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escriba su consulta jurídica..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Búsqueda Vectorial en la Galaxia de Datos
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
            
            # Generación de Respuesta (RAG)
            prompt_final = (
                f"Eres el SILC, experto en derecho mexicano. Analiza la siguiente consulta "
                f"con base en este contexto legal:\n\n{contexto}\n\nPregunta: {prompt}"
            )
            
            # Esta llamada ahora usará tu bono de $300 USD sin errores
            response = model.generate_content(prompt_final)
            
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            # Si el error persiste, este bloque nos dirá exactamente por qué
            st.error(f"Aviso del Sistema: {str(e)}")
