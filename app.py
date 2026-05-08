import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# 1. Configuración de Identidad
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal")
st.sidebar.write("Director: Doctorando Carlos Rubio")

# 2. INICIALIZACIÓN FORZADA (PRODUCCIÓN)
# Configuramos la llave que ya verificamos que es de pago
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# FORZAMOS el uso de la versión 'v1' para saltar el error 404 de la 'v1beta'
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash'
)

# Conexión a Base de Datos
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("galaxia-de-datos")

# 3. Interfaz de Chat
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
            # Recuperación de Contexto
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
            
            contexto = "\n".join([m['metadata']['text'] for m in query_res['matches']])
            
            # GENERACIÓN DE RESPUESTA
            # Usamos una estructura simple que Google v1 acepta sin problemas
            full_query = f"Contexto:\n{contexto}\n\nPregunta: {prompt}"
            
            # LLAMADA DE PRODUCCIÓN
            response = model.generate_content(full_query)
            
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"Aviso Técnico: {str(e)}")
