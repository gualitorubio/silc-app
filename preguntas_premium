import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from datetime import datetime
import pytz
from supabase import create_client, Client

# ==========================================
# 1. ARQUITECTURA DE MARCA E INTERFAZ PREMIUM
# ==========================================
st.set_page_config(
    page_title="SILC Consultas Premium - Rubio Intelligence Systems", 
    page_icon="💎", 
    layout="centered"
)

# Estilos CSS personalizados para el Plan Premium (Elegancia Corporativa y Oro)
st.markdown("""
<style> 
    .stImage {display: block; margin-left: auto; margin-right: auto;}
    .bloqueo-container {
        background-color: #ffffff; 
        padding: 35px; 
        border-radius: 15px; 
        border-left: 10px solid #C5A059;
        border-right: 2px solid #002D52;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .beneficio-premium {
        color: #002D52;
        font-weight: 600;
        margin-bottom: 8px;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ENLACE DINÁMICO (URL PARAMETERS)
# ==========================================
query_params = st.query_params
usuario_activo = query_params.get("user", None)

if not usuario_activo:
    st.error("❌ Acceso denegado. Sesión no identificada.")
    st.info("Por favor, inicia sesión desde tu panel corporativo en silcmexico.com.")
    st.stop()

# ==========================================
# 3. CONTROL ANTI-FRAUDE CDMX (100 CONSULTAS)
# ==========================================
zona_cdmx = pytz.timezone('America/Mexico_City')
ahora_cdmx = datetime.now(zona_cdmx)
fecha_actual_cdmx = ahora_cdmx.strftime('%Y-%m-%d')

try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)
except Exception as e:
    st.error("❌ Error de configuración crítica en la bóveda de seguridad.")
    st.stop()

def consultar_uso_db(email, fecha):
    try:
        res = supabase.table("consumo_diario").select("consultas").eq("email", email).eq("fecha", fecha).execute()
        if res.data:
            return res.data[0]["consultas"]
        return 0
    except Exception:
        st.error("⚠️ El servidor de validación no responde. Consultas pausadas por seguridad.")
        st.stop()

def actualizar_uso_db(email, fecha, nuevo_conteo):
    try:
        supabase.table("consumo_diario").upsert({
            "email": email, 
            "fecha": fecha, 
            "consultas": nuevo_conteo
        }).execute()
    except Exception:
        st.error("❌ No se pudo validar el cobro de tu consulta Premium.")
        st.stop()

conteo_real = consultar_uso_db(usuario_activo, fecha_actual_cdmx)
st.session_state.conteo_preguntas = conteo_real

# LÍMITE PREMIUM: 100 Consultas Diarias
LIMIT_PREGUNTAS = 100
preguntas_restantes = max(0, LIMIT_PREGUNTAS - st.session_state.conteo_preguntas)

# ==========================================
# 4. ENTORNO LATERAL (PANEL CORPORATIVO)
# ==========================================
with st.sidebar:
    try:
        st.image("SILC Logo.png", use_container_width=True)
        st.image("Rubio Intelligence Systems Logo.png", use_container_width=True)
    except:
        st.caption("SILC Premium - Rubio Intelligence Systems")
    
    st.divider()
    st.markdown("### 💎 Estatus Premium")
    st.info(f"**Socio:** {usuario_activo}\n\n**Módulo:** Consultas Corporativas\n\n**Plan:** Premium / Corporativo")
    
    if preguntas_restantes > 0:
        st.success(f"✅ Consultas hoy: {st.session_state.conteo_preguntas} de {LIMIT_PREGUNTAS}")
    else:
        st.error("❌ Límite diario Premium alcanzado")
        
    st.divider()
    st.markdown("### Dirección del Proyecto")
    st.write("Doctorando Carlos Rubio")
    st.caption("© 2026 Rubio Intelligence Systems")

# ==========================================
# 5. NÚCLEO DE LA APLICACIÓN Y RAG
# ==========================================
st.title("⚖️ SILC PREMIUM")
st.markdown("#### *Consultas jurídicas de alta precisión y profundidad histórica*")
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
# 6. COMPORTAMIENTO DEL PAYWALL PREMIUM
# ==========================================
if st.session_state.conteo_preguntas >= LIMIT_PREGUNTAS:
    st.markdown("""
        <div class="bloqueo-container">
            <h3 style="color: #002D52;">💎 Límite de socio Premium alcanzado (100/100)</h3>
            <p>Tu acceso se restablecerá a la medianoche (Hora CDMX). Como socio Premium tienes acceso a:</p>
            <div class="beneficio-premium">✔️ 100 Consultas diarias de Inteligencia Legal.</div>
            <div class="beneficio-premium">✔️ 30 Descargas diarias de escritos en Word (Word Generator activo).</div>
            <div class="beneficio-premium">✔️ Procesamiento prioritario en la Galaxia de Datos.</div>
            <p style="margin-top:20px;">Si tu firma requiere una capacidad superior o integración API personalizada, contacta a nuestra Dirección Técnica.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.link_button(
        "📞 CONTACTAR A SOPORTE CORPORATIVO", 
        url="mailto:soporte@silcmexico.com", 
        use_container_width=True,
        type="primary"
    )
else:
    if prompt := st.chat_input("Plantea tu caso jurídico de nivel corporativo..."):
        nuevo_conteo = st.session_state.conteo_preguntas + 1
        actualizar_uso_db(usuario_activo, fecha_actual_cdmx, nuevo_conteo)
        st.session_state.conteo_preguntas = nuevo_conteo
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Consultando la Galaxia de Datos en canal prioritario..."):
                    res_embed = pc.inference.embed(
                        model="multilingual-e5-large",
                        inputs=[prompt],
                        parameters={"input_type": "query"}
                    )
                    
                    query_res = index.query(
                        vector=res_embed[0].values, 
                        top_k=7, # MÁXIMA PROFUNDIDAD PARA PREMIUM
                        include_metadata=True,
                        namespace="silc-juridico"
                    )
                    
                    contexto_legal = "\n\n".join([str(m['metadata']['text']) for m in query_res['matches'] if 'metadata' in m and 'text' in m['metadata']])

                    instruccion = (
                        f"Eres el SILC (Sistema de Inteligencia Legal y Contexto) en su versión corporativa PREMIUM. "
                        f"Tu lema es 'Certeza jurídica con profundidad histórica'. "
                        f"Proporciona una asesoría exhaustiva, técnica y con alto rigor académico basándote en este contexto recuperado:\n\n"
                        f"{contexto_legal}\n\nPregunta del socio Premium: {prompt}"
                    )

                    response = model.generate_content(instruccion)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    st.rerun()

            except Exception as e:
                st.error(f"Error durante el procesamiento legal: {str(e)}")
