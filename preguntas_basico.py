import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from datetime import datetime
import pytz
from supabase import create_client, Client

# ==========================================
# 1. ARQUITECTURA DE MARCA E INTERFAZ
# ==========================================
st.set_page_config(
    page_title="SILC Consultas Básicas - Rubio Intelligence Systems", 
    page_icon="⚖️", 
    layout="centered"
)

# Estilos CSS personalizados para identidad visual y muro de pago
st.markdown("""
<style> 
    .stImage {display: block; margin-left: auto; margin-right: auto;}
    .bloqueo-container {
        background-color: #f8f9fa; 
        padding: 30px; 
        border-radius: 15px; 
        border-left: 8px solid #C5A059;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .beneficio-item {
        color: #1a1a1a;
        font-weight: 500;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ENLACE DINÁMICO CON BUILDERALL (URL PARAMETERS)
# ==========================================
query_params = st.query_params
# Extrae dinámicamente el correo enviado por la plataforma de Builderall en la variable '?user='
usuario_activo = query_params.get("user", None)

if not usuario_activo:
    st.error("❌ Acceso denegado. Sesión no identificada.")
    st.info("Por favor, inicia sesión desde tu panel de usuario protegido en silcmexico.com.")
    st.stop()

# ==========================================
# 3. CONTROL ANTI-FRAUDE CON BLINDAJE ULTRA (FAIL-CLOSED)
# ==========================================
zona_cdmx = pytz.timezone('America/Mexico_City')
ahora_cdmx = datetime.now(zona_cdmx)
fecha_actual_cdmx = ahora_cdmx.strftime('%Y-%m-%d')

try:
    # Conexión automática y segura mediante Secrets de Streamlit
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)
except Exception as e:
    st.error("❌ Error de configuración crítica en los microservicios de seguridad.")
    st.stop()

def consultar_uso_db(email, fecha):
    """
    Verifica el consumo del usuario dinámico en la BD. 
    Si el usuario no está registrado en Supabase, restringe el acceso o inicializa en 0.
    """
    try:
        res = supabase.table("consumo_diario").select("consultas").eq("email", email).eq("fecha", fecha).execute()
        if res.data:
            return res.data[0]["consultas"]
        return 0
    except Exception:
        st.error("⚠️ El servidor de validación no responde. Consultas pausadas por seguridad.")
        st.stop()

def actualizar_uso_db(email, fecha, nuevo_conteo):
    """
    Registra el consumo cobrando la consulta en Supabase (Upsert síncrono).
    """
    try:
        supabase.table("consumo_diario").upsert({
            "email": email, 
            "fecha": fecha, 
            "consultas": nuevo_conteo
        }).execute()
    except Exception:
        st.error("❌ No se pudo validar el cobro de tu consulta en el servidor central.")
        st.stop()

# Validación del usuario dinámico en tiempo de ejecución
conteo_real = consultar_uso_db(usuario_activo, fecha_actual_cdmx)
st.session_state.conteo_preguntas = conteo_real

LIMIT_PREGUNTAS = 5
preguntas_restantes = max(0, LIMIT_PREGUNTAS - st.session_state.conteo_preguntas)

# ==========================================
# 4. ENTORNO LATERAL (PANEL DE CONTROL)
# ==========================================
with st.sidebar:
    try:
        st.image("SILC Logo.png", use_container_width=True)
        st.image("Rubio Intelligence Systems Logo.png", use_container_width=True)
    except:
        st.caption("Logotipos corporativos vinculados.")
    
    st.divider()
    st.markdown("### 📊 Consumo de Créditos")
    st.info(f"**Usuario:** {usuario_activo}\n\n**Módulo:** Consultas Básicas\n\n**Plan:** Básico ($199.99/mes)")
    
    if preguntas_restantes > 0:
        st.success(f"⚡ Consultas hoy: {st.session_state.conteo_preguntas} de {LIMIT_PREGUNTAS}")
    else:
        st.error("❌ Límite diario alcanzado")
        
    st.divider()
    st.markdown("### Dirección del Proyecto")
    st.write("Doctorando Carlos Rubio")
    st.caption("© 2026 Rubio Intelligence Systems")

# ==========================================
# 5. NÚCLEO DE LA APLICACIÓN Y RAG
# ==========================================
st.title("⚖️ SILC")
st.markdown("#### *Certeza jurídica con profundidad histórica*")
st.divider()

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-3-flash-preview')
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("galaxia-de-datos")
except Exception as e:
    st.error(f"Error de conexión con microservicios: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 6. COMPORTAMIENTO DEL PAYWALL VS INPUT CHAT
# ==========================================
if st.session_state.conteo_preguntas >= LIMIT_PREGUNTAS:
    st.markdown(
        """
        <div class="bloqueo-container">
            <h3 style="color: #002D52;">⚠️ Has alcanzado tu límite de
