import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SILC - Rubio Intelligence Systems", page_icon="⚖️", layout="centered")

# --- ESTILO PERSONALIZADO PARA EL PILOTO ---
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1e3a8a; color: white; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #172554; color: #white; }
    .title-text { color: #1e3a8a; font-family: 'Helvetica Neue', sans-serif; }
    .footer { text-align: center; color: #64748b; font-size: 0.85em; margin-top: 60px; border-top: 1px solid #e2e8f0; padding-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- INICIALIZACIÓN SEGURA DE APIs (SECRETS) ---
try:
    PINECONE_KEY = st.secrets["PINECONE_API_KEY"]
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    
    genai.configure(api_key=GEMINI_KEY)
    pc = Pinecone(api_key=PINECONE_KEY)
    index = pc.Index("galaxia-de-datos")
    NAMESPACE = "silc-juridico"
    
    @st.cache_resource
    def load_embed_model():
        # Ejecución en CPU del servidor de Streamlit
        return SentenceTransformer('intfloat/multilingual-e5-large')
    
    embed_model = load_embed_model()
except Exception as e:
    st.error("⚠️ Error de configuración: Asegúrese de haber configurado los 'Secrets' en Streamlit Cloud.")

# --- ENCABEZADO INSTITUCIONAL ---
st.markdown("<h1 class='title-text'>⚖️ SILC</h1>", unsafe_allow_html=True)
st.markdown("### Sistema de Inteligencia Legal y Contexto")
st.caption("Certeza jurídica con profundidad histórica | Powered by **Rubio Intelligence Systems**")

st.markdown("---")

# --- ÁREA DE CONSULTA ---
st.markdown("#### **Plataforma de Consulta Jurídica (Piloto Universitario)**")
query = st.text_area("Describa el concepto jurídico o histórico que desea investigar:", 
                     placeholder="Ej: Compare el concepto de expropiación en 1930 vs la ley vigente...",
                     height=120)

if st.button("INICIAR ANÁLISIS JURÍDICO"):
    if query:
        with st.spinner("Analizando acervo de Rubio Intelligence Systems (91,077 registros)..."):
            try:
                # 1. Recuperación Vectorial
                query_emb = embed_model.encode(f"query: {query}").tolist()
                res = index.query(namespace=NAMESPACE, vector=query_emb, top_k=7, include_metadata=True)
                
                contexto_recuperado = ""
                for m in res['matches']:
                    meta = m['metadata']
                    contexto_recuperado += f"\n[DOCUMENTO: {meta['documento']} | AÑO: {meta['anio']} | ESTATUS: {meta['estatus']}]\n{meta['text']}\n"

                # 2. Instrucción Maestra (Golden Prompt)
                prompt_final = f"""
                Eres el SILC (Sistema de Inteligencia Legal y Contexto).
                Operado por: Rubio Intelligence Systems.
                Lema: "Certeza jurídica con profundidad histórica".

                Sustenta tu respuesta exclusivamente en este contexto recuperado:
                {contexto_recuperado}

                Pregunta del usuario: {query}

                Debes responder con estructura profesional:
                1. RESUMEN EJECUTIVO (Máximo 3 líneas).
                2. SUSTENTO NORMATIVO (Citas exactas con fuente y año).
                3. CONTEXTO Y EVOLUCIÓN (Análisis histórico).
                4. INTERPRETACIÓN ESTRATÉGICA (Aplicación práctica).
                """
                
                # 3. Generación con Gemini 1.5 Flash
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt_final)
                
                st.markdown("### **Resultado del Análisis**")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Error en el motor de análisis: {e}")
    else:
        st.warning("Por favor, ingrese una consulta para continuar.")

# --- PIE DE PÁGINA ---
st.markdown(f"""
    <div class='footer'>
        © 2026 Rubio Intelligence Systems | Dr. Carlos Rubio<br>
        Documentación Técnica: SILC V17.4.12.25<br>
        <i>Este sistema es una herramienta de investigación y no sustituye la asesoría legal profesional.</i>
    </div>
    """, unsafe_allow_html=True)
