# streamlit_app.py — Black hero + LEGAL AI PRO title ABOVE image + yellow chat UI

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# IMPORTANT: must be the first Streamlit command in the script
st.set_page_config(
    page_title="LegalAI Pro",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()

# ----------------------------
# CONFIG (EDIT IF NEEDED)
# ----------------------------
HERO_IMAGE_URL = "https://raw.githubusercontent.com/MoHiT05os/LegalAI-Pro-Fresh/main/s5uJ1E7t.jpg"

# ----------------------------

# CSS (BLACK + YELLOW THEME + TITLE ABOVE IMAGE)
# ----------------------------
st.markdown(
    """
<style>
/* Global background */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #000 !important;
    color: #f5f5f5 !important;
}

/* Hide Streamlit chrome */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"],
[data-testid="stSidebarNav"] {
    display: none !important;
}

/* Remove top padding so hero can be full-width */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
}

/* TITLE ABOVE IMAGE */
.title-section {
    width: 100%;
    text-align: center;
    margin-bottom: 1rem;
}

.hero-title {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    font-size: clamp(5rem, 10vw, 12rem);  /* Huge responsive size */
    font-weight: 900;  /* Ultra bold */
    background: linear-gradient(135deg, #facc15 0%, #eab308 55%, #ca8a04 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    text-shadow: 
        0 0 30px rgba(250, 204, 21, 0.9),
        0 6px 40px rgba(0,0,0,1);
    letter-spacing: 0.08em;
    margin: 0.5rem 0;
    line-height: 0.9;
}

/* Mobile fine-tuning */
@media (max-width: 768px) {
    .hero-title {
        font-size: clamp(4rem, 14vw, 8rem);
        letter-spacing: 0.04em;
    }
}

/* HERO IMAGE BELOW TITLE */
.hero-wrap {
    width: 100%;
    height: 50vh;
    background: #000;
    border-radius: 18px;
    overflow: hidden;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0.5rem 0 1.25rem 0;
    box-shadow: 0 30px 80px rgba(0,0,0,0.65);
    border: 1px solid rgba(250, 204, 21, 0.15);
}

/* subtle vignette so edges stay black */
.hero-wrap:before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at center,
        rgba(0,0,0,0.10) 0%,
        rgba(0,0,0,0.70) 70%,
        rgba(0,0,0,0.92) 100%);
    pointer-events: none;
    z-index: 2;
}

.hero-img {
    position: relative;
    z-index: 1;
    max-height: 50vh;
    width: auto;
    max-width: 98%;
    object-fit: contain;
    filter: drop-shadow(0 25px 60px rgba(0,0,0,0.75));
}

/* Chat bubbles */
.chat-message {
    padding: 1.4rem;
    border-radius: 20px;
    margin: 1rem 0;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    line-height: 1.55;
}

.user-message, .agent-message {
    background: linear-gradient(135deg, #facc15 0%, #eab308 55%, #ca8a04 100%);
    color: #0b0b0b !important;
    border: 1px solid rgba(0,0,0,0.25);
    box-shadow: 0 14px 40px rgba(250, 204, 21, 0.22);
}

/* Chat input */
.stChatInput {
    background: transparent !important;
}
.stChatInput > div {
    border-radius: 999px !important;
    border: 1px solid rgba(250, 204, 21, 0.35) !important;
    background: rgba(0,0,0,0.55) !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.6) !important;
}
.stChatInput input {
    color: #f5f5f5 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# RAG / QA CHAIN (unchanged logic)
# ----------------------------
from src.index import load_chroma
from src.prompts import PROMPT_TEMPLATE
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

@st.cache_resource
def build_qa_chain():
    db = load_chroma(collection_name="legal", persist_directory=Path.cwd() / "chroma_db")
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        input_key="query",
        output_key="result",
    )

def ask(question: str):
    qa = build_qa_chain()
    response = qa.invoke({"query": question})
    return response.get("result"), response.get("source_documents", [])

def clean_response(text: str) -> str:
    text = text or ""
    text = re.sub(r"[*_>`#]+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ----------------------------
# TITLE SECTION (ABOVE IMAGE)
# ----------------------------
st.markdown(
    """
<div class="title-section">
    <h1 class="hero-title">LEGAL AI PRO</h1>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# HERO IMAGE (BELOW TITLE)
# ----------------------------
st.markdown(
    f"""
<div class="hero-wrap">
    <img class="hero-img" src="{HERO_IMAGE_URL}" alt="Justice Statue">
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# CHAT
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "agent",
            "content": "Legal Agent Online! Ask me anything about BNS 2023, IPC, Evidence Act, or any Indian law.",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "agent":
            clean_content = clean_response(message["content"])
            st.markdown(
                f"""
<div class="chat-message agent-message">
    <strong>⚖️ Legal Agent</strong><br><br>
    {clean_content.replace("\n", "<br>")}
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div class="chat-message user-message">
    <strong>👤 You</strong><br><br>
    {message["content"]}
</div>
""",
                unsafe_allow_html=True,
            )

prompt = st.chat_input("Ask Legal Query...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(
            f"""
<div class="chat-message user-message">
    <strong>👤 You</strong><br><br>
    {prompt}
</div>
""",
            unsafe_allow_html=True,
        )

    with st.chat_message("agent"):
        with st.spinner("Analyzing..."):
            answer, _sources = ask(prompt)
            clean_answer = clean_response(answer)

        st.markdown(
            f"""
<div class="chat-message agent-message">
    <strong>⚖️ Legal Agent</strong><br><br>
    {clean_answer.replace("\n", "<br>")}
</div>
""",
            unsafe_allow_html=True,
        )

    st.session_state.messages.append({"role": "agent", "content": clean_answer})
