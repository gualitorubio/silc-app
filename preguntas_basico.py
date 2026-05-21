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
# 2. SISTEMA DE SEGURIDAD Y AJUSTE DE URL (FALLBACK CONTROL)
# ==========================================
query_params = st.query_params
# Corrección: Si el parámetro 'user' viene vacío o Safari lo oculta, se asigna gualitorubio@gmail.com por defecto para pruebas
usuario_activo = query_params.get("user", "gualitorubio@gmail.com")

if not usuario_activo:
    st.error("❌ Acceso denegado. Sesión no identificada.")
    st.info("Por favor, inicia sesión desde tu panel de usuario en silcmexico.com.")
    st.stop()

# ==========================================
# 3. CONTROL ANTI-FRAUDE CON BLINDAJE ULTRA (FAIL-CLOSED)
# ==========================================
zona_cdmx = pytz.timezone('America/Mexico_City')
ahora_cdmx = datetime.now(zona_cdmx)
fecha_actual_cdmx = ahora_cdmx.strftime('%Y-%m-%d')

try:
    # Conexión directa y veloz a la BD de control usando Secrets
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)
except Exception as e:
    st.error("❌ Error de configuración crítica en los microservicios de seguridad.")
    st.stop()

def consultar_uso_db(email, fecha):
    """
    Consulta el conteo real en Supabase. 
    Si la base de datos falla, se cierra por seguridad (Fail-Closed).
    """
    try:
        res = supabase.table("consumo_diario").select("consultas").eq("email", email).eq("fecha", fecha).execute()
        if res.data:
            return res.data[0]["consultas"]
        return 0
    except Exception:
        st.error("⚠️ El servidor de validación no responde. Consultas pausadas por seguridad.")
        st.info("Por favor, recarga la página en unos minutos o verifica tu conexión a internet.")
        st.stop()

def actualizar_uso_db(email, fecha, nuevo_conteo):
    """
    Registra el consumo de forma síncrona en Supabase (Upsert). 
    Si la actualización falla, detiene el proceso antes de quemar tokens de IA.
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

# Sincronización estricta en tiempo real en cada renderizado de pantalla
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
    
    # Corrección: Uso correcto de la biblioteca Pinecone actualizada sin el sufijo '-client'
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
            <h3 style="color: #002D52;">⚠️ Has alcanzado tu límite de 5 consultas diarias</h3>
            <p>Tu acceso se restablecerá automáticamente a la medianoche (Hora CDMX). Si necesitas continuar con tus investigaciones, actualiza al <b>Plan Profesional</b> por $899.99/mes:</p>
            <div class="beneficio-item">🚀 30 consultas diarias en el motor SILC de Inteligencia Legal.</div>
            <div class="beneficio-item">📄 10 descargas diarias de escritos en Word (Demandas, Amparos, Contratos).</div>
            <div class="beneficio-item">🔍 Buscador y Extractor de Clientes Potenciales B2B ilimitado.</div>
            <div class="beneficio-item">⚡ Servidores dedicados para una respuesta inmediata.</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.link_button(
        "⭐ MEJORAR AL PLAN PROFESIONAL", 
        url="https://silcmexico.com/sc-bridge/plans/VqyZwvp6Bxc4zAQd", 
        use_container_width=True,
        type="primary"
    )
else:
    if prompt := st.chat_input("Plantea tu interrogante o caso jurídico..."):
        # Cobro síncrono e inmediato en la BD de Supabase antes de procesar IA
        nuevo_conteo = st.session_state.conteo_preguntas + 1
        actualizar_uso_db(usuario_activo, fecha_actual_cdmx, nuevo_conteo)
        st.session_state.conteo_preguntas = nuevo_conteo
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Buscando fundamentos en la Galaxia de Datos..."):
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

                    instruccion = (
                        f"Eres el SILC (Sistema de Inteligencia Legal y Contexto). "
                        f"Tu lema es 'Certeza jurídica con profundidad histórica'. "
                        f"Analiza con rigor lo siguiente basándote en este contexto recuperado:\n\n"
                        f"{contexto_legal}\n\nPregunta del usuario: {prompt}"
                    )

                    response = model.generate_content(instruccion)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    
                    st.rerun()

            except Exception as e:
                st.error(f"Error durante el procesamiento legal: {str(e)}")
