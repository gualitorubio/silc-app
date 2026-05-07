import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 1. Configuración de la página
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️")

st.title("⚖️ SILC")
st.markdown("### Sistema de Inteligencia Legal y Contexto")
st.info("Desarrollado por Rubio Intelligence Systems | Dr. Carlos Rubio")

# 2. Cargar Secretos (Configurados en Streamlit Cloud)
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# 3. Inicializar Modelos y Clientes
@st.cache_resource
def load_models():
    # Configurar Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
    
    # Configurar Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("leyes-mexico") # Asegúrate que este sea el nombre de tu índice
    
    # Configurar Embeddings
    embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    return model_gemini, index, embed_model

try:
    gemini, index, embed_model = load_models()

    # 4. Interfaz de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escriba su consulta legal aquí..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # A. Generar Embedding de la pregunta
            query_vector = embed_model.encode(prompt).tolist()

            # B. Buscar en Pinecone
            search_results = index.query(vector=query_vector, top_k=7, include_metadata=True)
            
            # C. Construir Contexto
            context = "\n".join([res['metadata']['text'] for res in search_results['matches']])

            # D. Prompt Institucional para el Dr. Carlos Rubio
            full_prompt = f"""
            Eres el motor de inteligencia de SILC (Sistema de Inteligencia Legal y Contexto).
            Tu objetivo es asistir al Dr. Carlos Rubio y a la comunidad jurídica con respuestas precisas, 
            basadas estrictamente en la legislación mexicana y su contexto histórico.

            CONTEXTO LEGAL RECUPERADO:
            {context}

            PREGUNTA DEL USUARIO:
            {prompt}

            INSTRUCCIONES:
            1. Usa el contexto para responder. 
            2. Si la pregunta es histórica, contrasta la evolución de la ley.
            3. Si no encuentras la respuesta en el contexto, indícalo pero ofrece una interpretación basada en principios generales del derecho.
            4. Mantén un tono profesional, jurista y ejecutivo.
            """

            # E. Generar Respuesta con Gemini
            response = gemini.generate_content(full_prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

except Exception as e:
    st.error(f"Hubo un error en la conexión del sistema: {e}")
    st.info("Verifique que las API Keys en 'Secrets' y el nombre del índice en Pinecone sean correctos.")
