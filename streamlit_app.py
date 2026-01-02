# streamlit_app.py - SUPREME COURT AUTHORITY UI
import streamlit as st
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Imports (unchanged)
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
        llm=llm, chain_type="stuff", retriever=retriever, 
        return_source_documents=True, chain_type_kwargs={"prompt": prompt},
        input_key="query", output_key="result"
    )

def ask(question: str):
    qa = build_qa_chain()
    response = qa.invoke({"query": question})
    return response.get("result"), response.get("source_documents", [])

def clean_response(text):
    cleaned = re.sub(r'[*_>`#]+', '', text)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

# FULLSCREEN BLACK + HERO CONFIG
st.set_page_config(
    page_title="LegalAI Pro", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# SUPREME CSS - BLACK BG + PURPLE/YELLOW CHAT
st.markdown("""
<style>
    /* FULL BLACK BACKGROUND */
    .main { background: #000000 !important; padding: 0 !important; }
    .block-container { background: #000000 !important; padding: 2rem !important; }
    
    /* HIDE ALL STREAMLIT ELEMENTS */
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stStatusWidget"] { display: none !important; }
    section[data-testid="stSidebar"] > div > div > div > div > div > div > div > div > button { display: none !important; }
    
    /* HERO FULL WIDTH */
    .hero-section { width: 100vw !important; position: relative; margin: 0 !important; padding: 0 !important; }
    
    /* PURPLE/YELLOW CHAT BUBBLES */
    .user-message { 
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 50%, #d946ef 100%) !important; 
        color: white !important; 
        border: 1px solid #c084fc !important;
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.4) !important;
    }
    .agent-message { 
        background: linear-gradient(135deg, #facc15 0%, #eab308 50%, #ca8a04 100%) !important; 
        color: #000 !important; 
        border: 1px solid #f59e0b !important;
        box-shadow: 0 10px 30px rgba(250, 204, 21, 0.4) !important;
        font-weight: 500 !important;
    }
    
    /* Chat input purple */
    .stChatInput input { 
        background: linear-gradient(135deg, #8b5cf6, #a855f7) !important; 
        color: white !important; 
        border-radius: 25px !important; 
        border: 2px solid #c084fc !important;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3) !important;
    }
    
    /* Typography match statue */
    .chat-message { 
        padding: 1.8rem !important; 
        border-radius: 24px !important; 
        margin: 1.5rem 0 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        backdrop-filter: blur(12px) !important;
    }
</style>
""", unsafe_allow_html=True)

# FULL-WIDTH HERO JUSTICE IMAGE
st.markdown("""
<div class="hero-section" style="
    width: 100vw; 
    height: 60vh; 
    background: 
        linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.9)),
        url('https://raw.githubusercontent.com/MoHiT05os/LegalAI-Pro-Fresh/main/LEGAL-AI-PRO.jpg');
    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: -2rem -2rem 2rem -2rem !important;
    position: relative;
    border-radius: 0 !important;
">
    <!-- Optional overlay text -->
    <div style="
        text-align: center;
        color: white;
        text-shadow: 0 4px 20px rgba(0,0,0,0.8);
        z-index: 2;
    ">
        <h1 style="
            font-size: 5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #facc15 0%, #eab308 50%, #d97706 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            letter-spacing: -0.05em;
            font-family: 'Inter', -apple-system, sans-serif;
        ">
            LEGAL AI PRO
        </h1>
    </div>
</div>
""", unsafe_allow_html=True)

# CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "agent", "content": "⚖️ Legal Agent Online! Ask me anything about Bharatiya Nyaya Sanhita 2023, IPC, Evidence Act, or any Indian law. I'll cite exact sections + page numbers."}
    ]

# DISPLAY CONVERSATION (Purple/Yellow)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "agent":
            clean_content = clean_response(message['content'])
            st.markdown(f"""
                <div class="chat-message agent-message">
                    <strong>⚖️ Legal Agent</strong><br><br>
                    {clean_content.replace('\n', '<br>')}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You</strong><br>
                    {message['content']}
                </div>
            """, unsafe_allow_html=True)

# PURPLE CHAT INPUT
prompt = st.chat_input("💬 Ask Legal Query...", key="chat_input")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You</strong><br>
                {prompt}
            </div>
        """, unsafe_allow_html=True)

    with st.chat_message("agent"):
        with st.spinner("🔍 Supreme Court Analysis..."):
            answer, sources = ask(prompt)
            clean_answer = clean_response(answer)
            st.markdown(f"""
                <div class="chat-message agent-message">
                    <strong>⚖️ Legal Agent</strong><br><br>
                    {clean_answer.replace('\n', '<br>')}
                </div>
            """, unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "agent", "content": clean_answer})

# HIDE SIDEBAR COMPLETELY
# No sidebar code needed - fully hidden by CSS
