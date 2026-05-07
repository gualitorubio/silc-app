import streamlit as st
import google.generativeai as genai

st.title("⚖️ SILC: Rubio Intelligence Systems")

try:
    # Conexión directa y forzada
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-pro')
    
    if prompt := st.chat_input("Consulta jurídica..."):
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            # Generación inmediata para probar conexión
            response = model.generate_content(prompt)
            st.markdown(response.text)
except Exception as e:
    st.error(f"Error de comunicación con Google: {e}")
