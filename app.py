import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# ==========================================
# 1. CONFIGURACIÓN DE IDENTIDAD Y PÁGINA
# ==========================================
st.set_page_config(
    page_title="SILC - Rubio Intelligence Systems", 
    page_icon="⚖️", 
    layout="centered"
)

# Estilo para centrar el lema y logos si fuera necesario
st.markdown("<style> .stImage {display: block; margin-left: auto; margin-right: auto;} </style>", unsafe_allow_html=True)

# ==========================================
# 2. BARRA LATERAL (LOGOS E INSTRUCCIONES)
# ==========================================
with st.sidebar:
    # Carga de Logos
    try:
        st.image("SILC Logo.PNG", use_container_width=True)
        st.image("Rubio Intelligence Systems Logo.PNG", use_container_width=True)
    except:
        st.warning("Suba los archivos de imagen a GitHub para visualizar los logos.")
    
    st.divider()
    
    # Instrucciones de Uso
    st.markdown("### 📖 Instrucciones de Uso")
    st.info("""
    1. **Consulta:** Ingrese su duda o caso legal en el chat principal.
    2. **Análisis:** El sistema consultará la base de datos legislativa y convencional.
    3. **Rigor:** Obtendrá una respuesta fundamentada con perspectiva histórica y actual.
    """)
    
    st.divider()
    st.markdown("### Director")
    st.write("Doctorando Carlos Rubio")
    st.caption("© 2026 Rubio Intelligence Systems")

# ==========================================
# 3. CUERPO PRINCIPAL
# ==========================================
st.title("⚖️ SILC")
# LEMA ACTUALIZADO
st.markdown("#### *Certeza jurídica con profundidad histórica*")
st.divider()

# ==========================================
# 4. CONFIGURACIÓN DE MOTOR (IA & VECTOR DB)
# ==========================================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Usamos el modelo que ya confirmamos que funciona (Gemini 3)
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
except Exception as e:
    st.error(f"Error de configuración: {e}")

# Gestión del historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 5. PROCESAMIENTO DE CONSULTAS
# ==========================================
if prompt := st.chat_input("Realice su consulta jurídica..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Consultando registros del SILC..."):
                # Búsqueda en Pinecone
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
                
                contexto_legal = "\n\n".join([m['metadata']['text'] for m in query_res['matches']])

                # Generación de respuesta con el nuevo motor
                instruccion = (
                    f"Eres el SILC (Sistema de Inteligencia Legal y Contexto). "
                    f"Tu lema es 'Certeza jurídica con profundidad histórica'. "
                    f"Analiza con rigor lo siguiente basándote en este contexto:\n\n"
                    f"{contexto_legal}\n\nPregunta: {prompt}"
                )

                response = model.generate_content(instruccion)
                
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})

        except Exception as e:
            st.error(f"Aviso del Sistema: {str(e)}")
