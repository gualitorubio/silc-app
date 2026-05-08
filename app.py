import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# 1. Configuración de Identidad
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal")

# 2. Conexión de Pago (Forzando versión estable)
# La librería detectará tu API Key de pago automáticamente
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

# 3. Conexión a Base de Datos (Pinecone)
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("galaxia-de-datos")

# 4. Interfaz de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Consulta jurídica..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Búsqueda Vectorial
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
            # Pasamos el contexto directamente en el prompt para evitar errores de configuración
            prompt_final = f"Contexto legal:\n{contexto}\n\nPregunta: {prompt}\n\nResponde con rigor jurídico."
            
            response = model.generate_content(prompt_final)
            
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"Error Técnico: {str(e)}")
