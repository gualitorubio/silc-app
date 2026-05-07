import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# --- CONFIGURACIÓN DE IDENTIDAD ---
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")
st.title("⚖️ SILC: Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Dr. Carlos Rubio")

# --- CARGA DE RECURSOS ---
try:
    # Configuramos la IA con su llave de Google AI Studio
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Configuramos la Galaxia de Datos (Pinecone)
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
except Exception as e:
    st.error(f"Error en la configuración de llaves: {e}")

# --- INTERFAZ DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- LÓGICA DE PROCESAMIENTO JURÍDICO ---
if prompt := st.chat_input("Introduzca su consulta jurídica aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. Recuperación de Contexto (RAG)
            # Generamos el vector de la consulta para buscar en las 317 leyes
            res_embed = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query"}
            )
            
            # Buscamos en el índice de 1024 dimensiones
            search_results = index.query(
                vector=res_embed[0].values, 
                top_k=4, 
                include_metadata=True, 
                namespace="silc-juridico"
            )
            contexto_legal = "\n".join([r['metadata']['text'] for r in search_results['matches']])

            # 2. Generación de Respuesta con Fallback Automático
            # Si el modelo Flash falla por región, el sistema saltará al modelo Pro
            modelos_disponibles = ["gemini-1.5-flash", "gemini-pro"]
            respuesta_final = ""
            
            for nombre_modelo in modelos_disponibles:
                if not respuesta_final:
                    try:
                        model = genai.GenerativeModel(nombre_modelo)
                        chat_response = model.generate_content(
                            f"Actúa como el SILC. Analiza este contexto legal:\n{contexto_legal}\n\nPregunta: {prompt}"
                        )
                        respuesta_final = chat_response.text
                    except:
                        continue

            if respuesta_final:
                st.markdown(respuesta_final)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
            else:
                st.error("Lo sentimos, los servidores de Google aún están propagando su nueva clave. Intente de nuevo en un minuto.")

        except Exception as e:
            st.error(f"Error técnico en el motor SILC: {e}")
