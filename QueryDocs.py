import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import time
import base64

st.set_page_config(
    page_title="QueryDocs AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def img_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Orbitron:wght@700;900&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
footer, #MainMenu { visibility: hidden; }

/* ── Header ── */
.app-header {
    display: flex; align-items: center; gap: 14px;
    padding: 16px 0 14px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 20px;
    animation: fadeInDown 0.5s ease both;
}
.app-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem; font-weight: 900;
    color: #fff; letter-spacing: 2px;
}
.app-title span { color: #58a6ff; }
.app-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem; color: #8b949e;
    letter-spacing: 3px; margin-top: 3px;
}

/* ── Profile card in sidebar ── */
.profile-card {
    text-align: center;
    padding: 20px 0 16px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 16px;
    animation: fadeInDown 0.5s ease both;
}
.profile-avatar {
    width: 72px; height: 72px;
    border-radius: 50%;
    overflow: hidden;
    border: 2px solid #58a6ff;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.15),
                0 0 20px rgba(88,166,255,0.2);
    margin: 0 auto 10px;
    animation: pulse-av 2.5s ease-in-out infinite;
}
.profile-avatar img {
    width: 100%; height: 100%;
    object-fit: cover; border-radius: 50%;
}
@keyframes pulse-av {
    0%,100%{box-shadow:0 0 0 3px rgba(88,166,255,0.15),0 0 20px rgba(88,166,255,0.2);}
    50%    {box-shadow:0 0 0 4px rgba(88,166,255,0.3),0 0 30px rgba(88,166,255,0.35);}
}
.profile-name {
    font-family: 'Orbitron', monospace;
    font-size: 0.82rem; font-weight: 700;
    color: #fff; letter-spacing: 1px;
}
.profile-role {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.64rem; color: #58a6ff;
    letter-spacing: 2px; margin-top: 3px;
}
.profile-links {
    display: flex; justify-content: center;
    gap: 8px; margin-top: 10px; flex-wrap: wrap;
}
.p-link {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem; color: #8b949e !important;
    text-decoration: none !important;
    background: #0d1117;
    border: 1px solid #30363d;
    padding: 3px 10px; border-radius: 20px;
    transition: all 0.25s;
}
.p-link:hover { color: #58a6ff !important; border-color: #58a6ff; }

/* ── Upload zone ── */
.upload-card {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    transition: all 0.3s ease;
    animation: fadeInUp 0.5s ease both;
}
.upload-card:hover { border-color: #58a6ff; }
.upload-icon { font-size: 2.5rem; margin-bottom: 10px; }
.upload-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.95rem; color: #fff; margin-bottom: 6px;
}
.upload-sub { font-size: 0.85rem; color: #8b949e; }

/* ── PDF info banner ── */
.pdf-banner {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #58a6ff;
    border-radius: 8px;
    padding: 12px 18px;
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 16px;
    animation: fadeInUp 0.4s ease both;
}
.pdf-name {
    font-weight: 700; font-size: 0.95rem; color: #fff;
}
.pdf-meta {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem; color: #8b949e; margin-top: 2px;
}

/* ── Chat bubbles ── */
.chat-wrap { display: flex; flex-direction: column; gap: 16px; padding: 8px 0; }
.msg-user  { display: flex; justify-content: flex-end;  animation: slideInR 0.3s ease both; }
.msg-ai    { display: flex; justify-content: flex-start; animation: slideInL 0.3s ease both; }
.bubble-user {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: #fff; padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%; font-size: 0.97rem; line-height: 1.6;
    box-shadow: 0 4px 15px rgba(31,111,235,0.25);
}
.bubble-ai {
    background: #161b22; color: #c9d1d9;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    max-width: 80%; font-size: 0.95rem; line-height: 1.75;
    border: 1px solid #30363d;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.bubble-ai .answer-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem; color: #58a6ff;
    letter-spacing: 2px; margin-bottom: 8px;
    text-transform: uppercase;
}
.msg-meta {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem; color: #484f58;
    margin-top: 4px; padding: 0 6px;
}

/* ── Source chunks ── */
.sources-wrap {
    margin-top: 10px;
    border-top: 1px solid #21262d;
    padding-top: 10px;
}
.source-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem; color: #8b949e;
    letter-spacing: 2px; margin-bottom: 6px;
}
.source-chip {
    display: inline-block;
    background: rgba(88,166,255,0.08);
    border: 1px solid rgba(88,166,255,0.2);
    color: #58a6ff; padding: 3px 10px;
    border-radius: 4px; font-size: 0.72rem;
    font-family: 'Share Tech Mono', monospace;
    margin: 3px 3px; cursor: pointer;
}

/* ── Typing indicator ── */
.typing-wrap { display: flex; justify-content: flex-start; }
.typing-box {
    display: flex; align-items: center; gap: 5px;
    padding: 14px 18px; background: #161b22;
    border: 1px solid #30363d;
    border-radius: 18px 18px 18px 4px;
}
.t-dot {
    width: 7px; height: 7px; background: #58a6ff;
    border-radius: 50%;
    animation: tdot 1.2s ease-in-out infinite;
}
.t-dot:nth-child(2){animation-delay:0.2s;}
.t-dot:nth-child(3){animation-delay:0.4s;}
@keyframes tdot{0%,60%,100%{transform:translateY(0);opacity:0.4;}30%{transform:translateY(-8px);opacity:1;}}

/* ── Input ── */
.stTextInput > div > div > input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.97rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 12px rgba(88,166,255,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #21262d !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important; border-radius: 8px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.73rem !important; letter-spacing: 1px !important;
    transition: all 0.25s !important;
}
.stButton > button:hover {
    background: #30363d !important;
    border-color: #58a6ff !important; color: #58a6ff !important;
}
.send-btn > button {
    background: linear-gradient(135deg,#1f6feb,#388bfd) !important;
    color: #fff !important; border: none !important;
    box-shadow: 0 0 16px rgba(31,111,235,0.3) !important;
}
.send-btn > button:hover {
    box-shadow: 0 0 28px rgba(31,111,235,0.55) !important;
    transform: translateY(-2px) !important; color: #fff !important;
}

/* ── Stats ── */
.stat-row {
    display: flex; gap: 10px; flex-wrap: wrap;
    margin-bottom: 14px;
}
.stat-box {
    flex: 1; min-width: 80px;
    background: #161b22; border: 1px solid #21262d;
    border-top: 2px solid #58a6ff; border-radius: 6px;
    padding: 10px 12px; text-align: center;
}
.stat-num {
    font-family: 'Orbitron', monospace;
    font-size: 1.2rem; font-weight: 900; color: #58a6ff;
}
.stat-lbl {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem; color: #8b949e;
    letter-spacing: 1px; margin-top: 3px;
}

/* ── Welcome ── */
.welcome-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 30px;
    text-align: center; margin: 10px 0;
    animation: fadeInUp 0.6s ease both;
}
.wc-icon { font-size: 2.8rem; margin-bottom: 12px; }
.wc-title {
    font-family: 'Orbitron', monospace;
    font-size: 1rem; color: #fff; margin-bottom: 8px;
}
.wc-sub { font-size: 0.9rem; color: #8b949e; line-height: 1.65; }
.tip-chip {
    display: inline-block;
    background: rgba(88,166,255,0.08);
    border: 1px solid rgba(88,166,255,0.25);
    color: #58a6ff; padding: 4px 12px;
    border-radius: 20px; margin: 4px 3px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.67rem;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 2px dashed #30363d !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #58a6ff !important;
}

/* ── Glow divider ── */
.glow-div {
    border: none; height: 1px;
    background: linear-gradient(90deg,transparent,#58a6ff,transparent);
    margin: 16px 0; box-shadow: 0 0 8px rgba(88,166,255,0.3);
}

/* ── Animations ── */
@keyframes fadeInDown { from{opacity:0;transform:translateY(-16px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeInUp   { from{opacity:0;transform:translateY(16px)}  to{opacity:1;transform:translateY(0)} }
@keyframes slideInR   { from{opacity:0;transform:translateX(20px)}  to{opacity:1;transform:translateX(0)} }
@keyframes slideInL   { from{opacity:0;transform:translateX(-20px)} to{opacity:1;transform:translateX(0)} }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "pdf_name"    not in st.session_state: st.session_state.pdf_name    = None
if "pdf_pages"   not in st.session_state: st.session_state.pdf_pages   = 0
if "pdf_chunks"  not in st.session_state: st.session_state.pdf_chunks  = 0
if "q_count"     not in st.session_state: st.session_state.q_count     = 0

# ─── MODEL LOADERS ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource(show_spinner=False)
def load_llm():
    model_id  = "TinyLlama/TinyLlama-1.1B-chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=512, temperature=0.3, do_sample=True,
        pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1
    )
    return HuggingFacePipeline(pipeline=pipe)

# ─── PDF PROCESSOR ────────────────────────────────────────────────────────────
def process_pdf(uploaded_file):
    reader   = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    if not raw_text.strip():
        raise ValueError("No readable text found. PDF may be scanned/image-based.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_text(raw_text)
    embeddings  = load_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore, len(reader.pages), len(chunks)

# ─── ANSWER FUNCTION ──────────────────────────────────────────────────────────
def get_answer(question, vectorstore):
    retriever     = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)
    context       = "\n\n".join([f"---\n{doc.page_content}" for doc in relevant_docs])
    sources       = [doc.page_content[:120] + "..." for doc in relevant_docs]

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""<|system|>
You are QueryDocs AI, an intelligent document assistant. Use ONLY the context provided to answer the question clearly and accurately. If the answer is not in the context, say so honestly.
<|user|>
CONTEXT:
{context}

QUESTION:
{question}
<|assistant|>
"""
    )
    llm    = load_llm()
    chain  = prompt_template | llm
    result = chain.invoke({"context": context, "question": question})

    # extract only the assistant reply
    if "<|assistant|>" in result:
        answer = result.split("<|assistant|>")[-1].strip()
    else:
        answer = result.strip()

    return answer, sources

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Profile card
    img_b64 = img_to_base64("assets/NANII.png")
    avatar  = (f'<img src="data:image/png;base64,{img_b64}" alt="Sriram">'
               if img_b64 else
               '<div style="width:72px;height:72px;background:#1f6feb;border-radius:50%;margin:0 auto;"></div>')

    st.markdown(f"""
    <div class="profile-card">
        <div class="profile-avatar">{avatar}</div>
        <div class="profile-name">SRIRAM SAI</div>
        <div class="profile-role">// AI &amp; ML ENGINEER</div>
        <div class="profile-links">
            <a class="p-link" href="https://github.com/sriramsai18" target="_blank">💻 GitHub</a>
            <a class="p-link" href="https://www.linkedin.com/in/sriram-sai-laggisetti/" target="_blank">💼 LinkedIn</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # PDF upload
    st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;color:#8b949e;letter-spacing:2px;margin-bottom:8px;">📄 UPLOAD DOCUMENT</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "", type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("🔍 Processing PDF..."):
                try:
                    vs, pages, chunks = process_pdf(uploaded_file)
                    st.session_state.vectorstore = vs
                    st.session_state.pdf_name    = uploaded_file.name
                    st.session_state.pdf_pages   = pages
                    st.session_state.pdf_chunks  = chunks
                    st.session_state.messages    = []
                    st.session_state.q_count     = 0
                    st.success("✅ PDF ready!")
                except Exception as e:
                    st.error(f"❌ {str(e)}")

    st.markdown("---")

    # Stats
    if st.session_state.pdf_name:
        st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#8b949e;margin-bottom:10px;letter-spacing:2px;">📊 DOCUMENT STATS</div>
        <div style="display:flex;flex-direction:column;gap:6px;">
            <div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:8px 12px;font-family:'Share Tech Mono',monospace;font-size:0.7rem;">
                📄 <span style="color:#c9d1d9;">{st.session_state.pdf_name[:22]}{'...' if len(st.session_state.pdf_name)>22 else ''}</span>
            </div>
            <div style="display:flex;gap:6px;">
                <div style="flex:1;background:#0d1117;border:1px solid #21262d;border-top:2px solid #58a6ff;border-radius:6px;padding:8px;text-align:center;">
                    <div style="font-family:'Orbitron',monospace;font-size:1rem;color:#58a6ff;">{st.session_state.pdf_pages}</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#8b949e;">PAGES</div>
                </div>
                <div style="flex:1;background:#0d1117;border:1px solid #21262d;border-top:2px solid #58a6ff;border-radius:6px;padding:8px;text-align:center;">
                    <div style="font-family:'Orbitron',monospace;font-size:1rem;color:#58a6ff;">{st.session_state.pdf_chunks}</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#8b949e;">CHUNKS</div>
                </div>
                <div style="flex:1;background:#0d1117;border:1px solid #21262d;border-top:2px solid #58a6ff;border-radius:6px;padding:8px;text-align:center;">
                    <div style="font-family:'Orbitron',monospace;font-size:1rem;color:#58a6ff;">{st.session_state.q_count}</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#8b949e;">ASKED</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

    # Clear button
    if st.button("🗑️ CLEAR CHAT", use_container_width=True):
        st.session_state.messages = []
        st.session_state.q_count  = 0
        st.rerun()

    # New PDF button
    if st.button("📄 LOAD NEW PDF", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.pdf_name    = None
        st.session_state.pdf_pages   = 0
        st.session_state.pdf_chunks  = 0
        st.session_state.messages    = []
        st.session_state.q_count     = 0
        st.rerun()

# ─── MAIN AREA ────────────────────────────────────────────────────────────────
# Header
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">QUERY<span>DOCS</span> AI 📚</div>
        <div class="app-sub">// INTELLIGENT DOCUMENT Q&amp;A · RAG PIPELINE · TINYLLAMA 1.1B</div>
    </div>
</div>
""", unsafe_allow_html=True)

# PDF banner (when loaded)
if st.session_state.pdf_name:
    st.markdown(f"""
    <div class="pdf-banner">
        <span style="font-size:1.4rem;">📄</span>
        <div>
            <div class="pdf-name">{st.session_state.pdf_name}</div>
            <div class="pdf-meta">{st.session_state.pdf_pages} pages · {st.session_state.pdf_chunks} chunks · ready to query</div>
        </div>
        <span style="margin-left:auto;background:rgba(88,166,255,0.1);border:1px solid rgba(88,166,255,0.3);
              color:#58a6ff;padding:4px 12px;border-radius:20px;
              font-family:'Share Tech Mono',monospace;font-size:0.65rem;">● ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)

# Chat area
if not st.session_state.vectorstore:
    st.markdown("""
    <div class="welcome-card">
        <div class="wc-icon">📚</div>
        <div class="wc-title">WELCOME TO QUERYDOCS AI</div>
        <div class="wc-sub">
            Upload any PDF document from the sidebar and start asking questions.<br>
            Powered by RAG pipeline + TinyLlama 1.1B for accurate, context-aware answers.
        </div>
        <br>
        <span class="tip-chip">📋 Legal documents</span>
        <span class="tip-chip">📊 Research papers</span>
        <span class="tip-chip">📖 Study material</span>
        <span class="tip-chip">📝 Reports</span>
    </div>
    """, unsafe_allow_html=True)

else:
    # Chat history
    if st.session_state.messages:
        chat_html = '<div class="chat-wrap">'
        for msg in st.session_state.messages:
            ts = msg.get("time", "")
            if msg["role"] == "user":
                chat_html += f"""
                <div class="msg-user">
                    <div>
                        <div class="bubble-user">{msg["content"]}</div>
                        <div class="msg-meta" style="text-align:right;">YOU · {ts}</div>
                    </div>
                </div>"""
            else:
                sources_html = ""
                if msg.get("sources"):
                    chips = "".join(f'<span class="source-chip">📎 Chunk {i+1}</span>'
                                    for i, _ in enumerate(msg["sources"]))
                    sources_html = f"""
                    <div class="sources-wrap">
                        <div class="source-label">// SOURCE CHUNKS USED</div>
                        {chips}
                    </div>"""
                chat_html += f"""
                <div class="msg-ai">
                    <div>
                        <div class="bubble-ai">
                            <div class="answer-label">// QUERYDOCS RESPONSE</div>
                            {msg["content"]}
                            {sources_html}
                        </div>
                        <div class="msg-meta">📚 QUERYDOCS AI · {ts} · {msg.get("elapsed","?")}s</div>
                    </div>
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="welcome-card" style="padding:20px;">
            <div style="font-size:1.6rem;margin-bottom:8px;">💬</div>
            <div class="wc-title" style="font-size:0.85rem;">DOCUMENT LOADED — START ASKING</div>
            <div class="wc-sub" style="font-size:0.82rem;">Ask anything about the uploaded document.</div>
            <br>
            <span class="tip-chip">💡 Summarize this document</span>
            <span class="tip-chip">💡 What are the key findings?</span>
            <span class="tip-chip">💡 List all important dates</span>
        </div>
        """, unsafe_allow_html=True)

    # Input row
    st.markdown('<hr class="glow-div">', unsafe_allow_html=True)
    col_q, col_btn = st.columns((5, 1))

    with col_q:
        question = st.text_input(
            "", placeholder="// ask a question about your document...",
            label_visibility="collapsed", key="question_input"
        )
    with col_btn:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        ask_btn = st.button("▶ ASK", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Generate answer
    if (ask_btn or question) and question.strip():
        ts = time.strftime("%H:%M")
        st.session_state.messages.append({
            "role": "user", "content": question.strip(), "time": ts
        })
        st.session_state.q_count += 1

        # typing indicator
        typing_slot = st.empty()
        typing_slot.markdown("""
        <div class="typing-wrap">
            <div class="typing-box">
                <div class="t-dot"></div>
                <div class="t-dot"></div>
                <div class="t-dot"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            start   = time.time()
            answer, sources = get_answer(question.strip(), st.session_state.vectorstore)
            elapsed = round(time.time() - start, 1)

            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "sources": sources,
                "time":    time.strftime("%H:%M"),
                "elapsed": elapsed
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error generating answer: {str(e)}",
                "sources": [], "time": time.strftime("%H:%M"), "elapsed": 0
            })

        typing_slot.empty()
        st.rerun()
