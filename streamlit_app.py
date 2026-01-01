# streamlit_app.py - PREMIUM LEGAL CHATBOT UI (FIXED TYPOs)
import streamlit as st
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# YOUR EXACT ask_cli CODE (NO CHANGES)
from src.index import load_chroma
from src.prompts import PROMPT_TEMPLATE
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# HIDE GITHUB UI ELEMENTS
st.markdown("""
    <style>
        /* Hide GitHub fork/star buttons */
        section[data-testid="stSidebar"] > div > div > div > div > div > div > div > div > button {
            display: none !important;
        }
        
        /* Hide Streamlit header/menu */
        [data-testid="stHeader"] { display: none !important; }
        [data-testid="stSidebarNav"] { display: none !important; }
        
        /* Clean professional look */
        .main .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)


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
    """Remove ALL markdown artifacts for clean display"""
    cleaned = re.sub(r'[*_>`#]+', '', text)  # Remove **bold**, *italic*, `code`, headers
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Fix extra newlines
    return cleaned.strip()

# 🔥 COOL CHATBOT UI
st.set_page_config(
    page_title="LegalAI Pro", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .chat-message {padding: 1.5rem; border-radius: 20px; margin: 1rem 0;}
    .user-message {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;}
    .agent-message {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);}
    .input-box {border-radius: 25px !important; border: 2px solid #e1e5e9;}
    </style>
""", unsafe_allow_html=True)

# HEADER - FIXED (No raw </div>)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <div style="
    text-align: center; 
    padding: 4rem 2rem 2.5rem 2rem;
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 50%, #1e3a8a 100%);
    border-radius: 24px;
    margin: 0 -1rem 2rem -1rem;
    box-shadow: 0 25px 50px -12px rgba(30, 64, 175, 0.4);
    border: 1px solid rgba(251, 191, 36, 0.4);
    backdrop-filter: blur(20px);
">
    <!-- MAIN TITLE - Elegant Gold -->
    <h1 style="
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #facc15 0%, #eab308 50%, #ca8a04 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 1rem 0;
        letter-spacing: -0.03em;
        text-shadow: 0 4px 20px rgba(250, 204, 21, 0.4);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    ">
        LegalAI Pro
    </h1>
</div>
    """, unsafe_allow_html=True)

# CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "agent", "content": "👋 Legal Agent Online! Ask me anything about Bharatiya Nyaya Sanhita 2023, IPC, Evidence Act, or any Indian law. I'll cite exact sections + page numbers."}
    ]

# DISPLAY CONVERSATION
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "agent":
            clean_content = clean_response(message['content'])
            st.markdown(f"""
                <div class="chat-message agent-message">
                    <strong>⚖️ Legal Agent</strong><br><br>
                    {clean_content.replace('\\n', '<br>')}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You</strong><br>
                    {message['content']}
                </div>
            """, unsafe_allow_html=True)

# CENTERED PROMPT INPUT
prompt = st.chat_input(
    "💬 Type your queries here!", 
    key="chat_input"
)

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You</strong><br>
                {prompt}
            </div>
        """, unsafe_allow_html=True)

    # Generate response
    with st.chat_message("agent"):
        with st.spinner("🔍 Legal Agent analyzing BNS 2023 & IPC..."):
            answer, sources = ask(prompt)
            clean_answer = clean_response(answer)
            
            response_html = f"""
                <div class="chat-message agent-message">
                    <strong>⚖️ Legal Agent</strong><br><br>
                    {clean_answer.replace('\\n', '<br>')}
                </div>
            """
            
            st.markdown(response_html, unsafe_allow_html=True)
        
        # Store CLEANED answer in history (no markdown artifacts)
        st.session_state.messages.append({"role": "agent", "content": clean_answer})

# SIDEBAR - Demo Prompts
with st.sidebar:
    st.markdown("## 🚀 Quick Demos")
    demo_prompts = [
        "BNS 2023: Director took ₹75L crypto scam money and fled",
        "What is BNS Section 318(4) punishment?",
        "IPC 420 vs BNS 318 differences",
        "BNS murder punishment Section?",
        "Evidence Act cross-examination rules"
    ]
    
    st.markdown("---")
    for i, demo in enumerate(demo_prompts):
        if st.button(f"💡 {demo[:50]}...", key=f"demo_{i}"):
            st.session_state.messages.append({"role": "user", "content": demo})
            st.rerun()
    
    st.markdown("---")
    st.markdown("[⭐ Deployed on Streamlit](https://share.streamlit.io)")
