import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

st.set_page_config(
    page_title="RAG · Analiza tu PDF",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Space+Grotesk:wght@400;500;600&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #dff0ff 0%, #f0f8ff 40%, #e8f4fd 70%, #d4ecff 100%) !important;
    font-family: 'Nunito', sans-serif;
    color: #1a3a5c;
}

/* Cloud SVG background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(ellipse 220px 90px at 8% 18%, rgba(255,255,255,0.85) 0%, transparent 70%),
        radial-gradient(ellipse 180px 70px at 15% 22%, rgba(255,255,255,0.7) 0%, transparent 65%),
        radial-gradient(ellipse 300px 110px at 75% 8%, rgba(255,255,255,0.9) 0%, transparent 70%),
        radial-gradient(ellipse 200px 80px at 85% 14%, rgba(255,255,255,0.75) 0%, transparent 65%),
        radial-gradient(ellipse 250px 95px at 45% 90%, rgba(255,255,255,0.7) 0%, transparent 70%),
        radial-gradient(ellipse 160px 60px at 60% 85%, rgba(255,255,255,0.6) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    max-width: 740px !important;
    padding: 2rem 2rem 4rem !important;
    position: relative;
    z-index: 1;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a6dbf 0%, #0f4f9e 100%) !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * {
    color: #e8f4ff !important;
    font-family: 'Nunito', sans-serif !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.9rem;
    line-height: 1.65;
    color: #b8d8f8 !important;
}
[data-testid="stSidebarNav"] { display: none; }

/* ── Header card ── */
.sky-header {
    background: linear-gradient(135deg, #1a6dbf 0%, #0f8fe8 60%, #38b5ff 100%);
    border-radius: 20px;
    padding: 2rem 2.2rem 1.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(15, 90, 200, 0.25), 0 2px 8px rgba(15, 90, 200, 0.15);
}
.sky-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        radial-gradient(ellipse 180px 65px at 80% 30%, rgba(255,255,255,0.35) 0%, transparent 65%),
        radial-gradient(ellipse 120px 45px at 88% 20%, rgba(255,255,255,0.25) 0%, transparent 55%),
        radial-gradient(ellipse 200px 70px at 10% 70%, rgba(255,255,255,0.2) 0%, transparent 65%);
    pointer-events: none;
}
.sky-header h1 {
    font-family: 'Nunito', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.4rem;
    position: relative;
    text-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.sky-header p {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.8);
    margin: 0;
    position: relative;
    font-weight: 500;
}
.sky-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    backdrop-filter: blur(4px);
}

/* ── Cards / panels ── */
.sky-card {
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(100,170,255,0.3);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(15, 90, 180, 0.08);
}
.card-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a9fd4;
    margin-bottom: 0.6rem;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.9) !important;
    border: 1.5px solid rgba(100,170,255,0.45) !important;
    border-radius: 10px !important;
    color: #1a3a5c !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #1a6dbf !important;
    box-shadow: 0 0 0 3px rgba(26,109,191,0.12) !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label { display: none !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.65) !important;
    border: 2px dashed rgba(56,181,255,0.5) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(255,255,255,0.85) !important;
    border-color: #1a6dbf !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploaderDropzone"] > div {
    color: #5a9abf !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1a6dbf 0%, #0f8fe8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 0.65rem 2.2rem !important;
    box-shadow: 0 4px 16px rgba(15, 110, 200, 0.3) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    width: 100%;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(15, 110, 200, 0.38) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.87rem !important;
    border: none !important;
}
.stSuccess {
    background: rgba(220, 245, 235, 0.9) !important;
    color: #0d6e3f !important;
}
.stInfo {
    background: rgba(220, 238, 255, 0.9) !important;
    color: #0e4f8a !important;
}
.stWarning {
    background: rgba(255, 243, 215, 0.9) !important;
    color: #7a4f00 !important;
}
.stError {
    background: rgba(255, 230, 228, 0.9) !important;
    color: #8a1500 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.85rem !important;
    color: #4a80bf !important;
}

/* ── Answer box ── */
.answer-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.92) 0%, rgba(240,250,255,0.95) 100%);
    border: 1.5px solid rgba(56,181,255,0.35);
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    margin-top: 1rem;
    box-shadow: 0 6px 28px rgba(15,90,180,0.1);
    font-family: 'Nunito', sans-serif;
    font-size: 0.95rem;
    line-height: 1.75;
    color: #1a3a5c;
    position: relative;
}
.answer-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #1a6dbf, #38b5ff);
    border-radius: 16px 16px 0 0;
}
.answer-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1a6dbf;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.answer-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(56,181,255,0.3);
}

/* ── Progress / stats chips ── */
.stat-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 0.6rem 0;
}
.stat-chip {
    background: rgba(26,109,191,0.1);
    border: 1px solid rgba(26,109,191,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #1a5a9e;
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Footer ── */
.sky-footer {
    text-align: center;
    font-size: 0.72rem;
    color: #8ab8d8;
    margin-top: 2.5rem;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1rem;'>
        <div style='font-size:1.5rem; margin-bottom:0.5rem;'>☁️</div>
        <div style='font-size:1.1rem; font-weight:800; color:#fff; margin-bottom:0.8rem;'>¿Cómo funciona?</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    **1. Ingresa tu API Key** de OpenAI para activar el análisis.

    **2. Sube tu PDF** — el documento se procesará y dividirá en fragmentos para búsqueda semántica.

    **3. Haz tu pregunta** — el sistema buscará los fragmentos más relevantes y generará una respuesta con GPT.

    ---
    Este método se llama **RAG** (Retrieval-Augmented Generation): combina búsqueda vectorial con generación de lenguaje.
    """)
    st.markdown(f"""
    <div style='margin-top:1.5rem; padding: 0.75rem 1rem; background: rgba(255,255,255,0.1);
    border-radius:10px; font-size:0.75rem; color:rgba(255,255,255,0.6);'>
    Python {platform.python_version()}
    </div>
    """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sky-header">
    <div class="sky-badge">✦ RAG · Análisis de Documentos</div>
    <h1>Habla con tu PDF ☁️</h1>
    <p>Sube un documento y haz cualquier pregunta — la IA encontrará la respuesta.</p>
</div>
""", unsafe_allow_html=True)

# ── API Key ───────────────────────────────────────────────────────────────────
st.markdown('<div class="sky-card"><div class="card-label">🔑 API Key de OpenAI</div>', unsafe_allow_html=True)
ke = st.text_input("key", placeholder="sk-...", type="password", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Ingresa tu clave de API de OpenAI para continuar.")

# ── PDF Upload ────────────────────────────────────────────────────────────────
st.markdown('<div class="sky-card"><div class="card-label">📄 Documento PDF</div>', unsafe_allow_html=True)
pdf = st.file_uploader("pdf", type="pdf", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ── Process ───────────────────────────────────────────────────────────────────
if pdf is not None and ke:
    try:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=500, chunk_overlap=20, length_function=len
        )
        chunks = text_splitter.split_text(text)

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-chip">📖 {len(pdf_reader.pages)} páginas</div>
            <div class="stat-chip">✂️ {len(chunks)} fragmentos</div>
            <div class="stat-chip">🔤 {len(text):,} caracteres</div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Creando base de conocimiento vectorial..."):
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.success("¡Documento listo! Ya puedes hacer preguntas.")

        st.markdown('<div class="sky-card"><div class="card-label">💬 Tu pregunta</div>', unsafe_allow_html=True)
        user_question = st.text_area(
            "q", placeholder="Ej: ¿Cuáles son los puntos principales del documento?",
            label_visibility="collapsed", height=100
        )
        ask_btn = st.button("→ Obtener respuesta")
        st.markdown('</div>', unsafe_allow_html=True)

        if user_question and ask_btn:
            with st.spinner("Buscando en el documento..."):
                docs = knowledge_base.similarity_search(user_question)
                llm = OpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)

            st.markdown('<div class="answer-label">Respuesta</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card">{response}</div>', unsafe_allow_html=True)

    except Exception as e:
        import traceback
        st.error(f"Error al procesar el PDF: {str(e)}")
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Ingresa tu clave de API de OpenAI para procesar el documento.")
else:
    st.info("Sube un archivo PDF para comenzar.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sky-footer">RAG · Retrieval-Augmented Generation · OpenAI + FAISS + LangChain</div>
""", unsafe_allow_html=True)
