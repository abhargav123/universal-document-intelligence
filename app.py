import streamlit as st
import os
import re
from collections import Counter
import docx2txt
import urllib.request
import urllib.parse
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Base configuration setting layout width
st.set_page_config(page_title="Universal Doc AI Engine", layout="wide")

# High-Contrast Premium Tech Theme CSS Override with Heavy Specificity Tags
st.markdown("""
    <style>
    .stApp {
        background-color: #0b0c10 !important;
        color: #ffffff !important;
    }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    p, span, label, li, div, h1, h2, h3, h4 {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #14161d !important;
        border-right: 1px solid #282c37;
    }
    .dev-profile-card {
        background-color: #14161d;
        padding: 25px;
        border-radius: 8px;
        border: 1px solid #282c37;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    div[data-testid="stMetricValue"] {
        color: #00ffd0 !important;
        font-weight: 700 !important;
        font-size: 26px !important;
    }
    div[data-testid="metric-container"] {
        background-color: #14161d !important;
        border: 1px solid #282c37 !important;
        border-radius: 8px;
        padding: 15px;
    }
    [data-testid="stFileUploader"] {
        background-color: #14161d !important;
        border: 1px solid #282c37 !important;
        padding: 10px !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1a1d26 !important;
        border: 2px dashed #454e6d !important;
        border-radius: 6px !important;
    }
    [data-testid="stFileUploaderDropzone"] *, .stFileUploader * {
        color: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #282c37 !important;
        border: 1px solid #454e6d !important;
        color: #ffffff !important;
    }
    .streamlit-expanderHeader {
        background-color: #14161d !important;
        border: 1px solid #282c37 !important;
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Global Constants for Developer info
DEV_NAME = "Anupam Bhargav"
UNIVERSITY = "Madan Mohan Malaviya University of Technology"
DEPARTMENT = "Computer Science & Engineering Department"
DEV_EMAIL = "abhargav3637@gmail.com"
DEV_PHONE = "+91 8081300874"

st.title("Universal Document Intelligence Engine")
st.caption("Enterprise Hybrid RAG System with Real-Time Web Search & Core Local Document Context.")
st.markdown("---")

# Initialize Global Session States
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "doc_summaries" not in st.session_state:
    st.session_state.doc_summaries = {}
if "per_doc_history" not in st.session_state:
    st.session_state.per_doc_history = {}
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}

if not os.path.exists("extracted_pdf"):
    os.makedirs("extracted_pdf")

@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

embeddings = load_system()

# Native Core REST Request Component for Groq Cloud API with Cloudflare Headers
def invoke_groq_llm(prompt):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY environment configuration variable missing in Advanced Settings."
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        
        # Adding standard browser headers to bypass Cloudflare Error 1010
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as he:
        err_res = he.read().decode("utf-8")
        return f"Enterprise Inference Pipeline Offline: HTTP Error {he.code}. Details: {err_res}"
    except Exception as e:
        return f"Enterprise Inference Pipeline Offline: {str(e)}"

def search_live_web(query):
    try:
        enc_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={enc_query}&format=json&no_html=1"
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
        web_context = ""
        if data.get("AbstractText"):
            web_context += f"Source: DuckDuckGo Encyclopedia - {data['AbstractText']}\n"
        
        if data.get("RelatedTopics"):
            topics = [t['Text'] for t in data['RelatedTopics'] if 'Text' in t][:2]
            if topics:
                web_context += f"Related Semantics - " + " | ".join(topics)
                
        if web_context:
            return web_context
        return "No direct public encyclopedia facts found for this specific query."
    except Exception:
        return "Web intelligence layer network bridge temporary offline."

def mask_sensitive_data(text):
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\b\d{10}\b|\b\d{3}[-\s]\d{3}[-\s]\d{4}\b', '[REDACTED_CONTACT]', text)
    return text

def analyze_text_metadata(text_content):
    words = re.findall(r'\b\w+\b', text_content.lower())
    total_words = len(words)
    stopwords = set(["the", "a", "and", "of", "to", "in", "is", "for", "on", "with", "an", "this", "by", "that", "at", "it", "from", "as", "your", "our"])
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    top_keywords = [item[0] for item in Counter(filtered_words).most_common(5)]
    est_reading_time = max(1, round(total_words / 200))
    return {"words": total_words, "keywords": top_keywords, "time": est_reading_time}

def extract_text_by_format(file_path, file_name, mask_enabled):
    ext = file_name.split(".")[-1].lower()
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if mask_enabled:
            for d in docs:
                d.page_content = mask_sensitive_data(d.page_content)
        chunks = text_splitter.split_documents(docs)
        raw_text = " ".join([doc.page_content for doc in docs])
    elif ext == "docx":
        text = docx2txt.process(file_path)
        if mask_enabled:
            text = mask_sensitive_data(text)
        doc = Document(page_content=text, metadata={"source": file_name})
        chunks = text_splitter.split_documents([doc])
        raw_text = text
    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if mask_enabled:
            text = mask_sensitive_data(text)
        doc = Document(page_content=text, metadata={"source": file_name})
        chunks = text_splitter.split_documents([doc])
        raw_text = text
    else:
        st.sidebar.error(f"Unsupported format: .{ext}")
        return None, None
    return chunks, raw_text

# ================= SIDEBAR: CONTROL CENTER =================
st.sidebar.header("📁 Document Control Center")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, or TXT documents", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

st.sidebar.subheader("🛡️ Privacy & Compliance")
mask_data = st.sidebar.toggle("Enable PII Data Masking", value=False)

active_docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        active_docs.append(uploaded_file.name)
        file_path = os.path.join("extracted_pdf", uploaded_file.name)
        
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        if uploaded_file.name not in st.session_state.vector_stores:
            with st.sidebar.spinner(f"Processing {uploaded_file.name}..."):
                chunks, raw_text = extract_text_by_format(file_path, uploaded_file.name, mask_data)
                if chunks:
                    st.session_state.vector_stores[uploaded_file.name] = Chroma.from_documents(chunks, embeddings)
                    st.session_state.per_doc_history[uploaded_file.name] = []
                    st.session_state.doc_stats[uploaded_file.name] = analyze_text_metadata(raw_text)
                    
                    summary_prompt = (
                        f"Analyze this document text and provide a concise 2-sentence summary followed by 3 bulleted key insights.\n"
                        f"Text:\n{raw_text[:3000]}\n\nResponse:"
                    )
                    st.session_state.doc_summaries[uploaded_file.name] = invoke_groq_llm(summary_prompt)

selected_doc = None
if active_docs:
    st.sidebar.subheader("🎯 Active Context Selection")
    selected_doc = st.sidebar.selectbox("Choose document to chat with:", active_docs)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 AI Core Modality")
    enable_web = st.sidebar.toggle("Enable Live Web Search (Hybrid RAG)", value=False)
    
    if st.sidebar.button("Clear Memory for this Document"):
        st.session_state.per_doc_history[selected_doc] = []
        st.rerun()
else:
    st.info("System Engine Online. Please upload and select a document from the sidebar to activate workspace dashboards.")

# ================= MAIN AREA: ACTIVE WORKSPACE =================
if selected_doc:
    st.subheader(f"📊 Insights Dashboard: {selected_doc}")
    
    stats = st.session_state.doc_stats[selected_doc]
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Word Count", value=f"{stats['words']} words")
    col2.metric(label="Estimated Reading Time", value=f"{stats['time']} min")
    col3.metric(label="Core Topic Keywords", value=", ".join(stats['keywords']).title())
    
    with st.expander("📝 Document Executive Summary & Key Takeaways", expanded=True):
        st.markdown(st.session_state.doc_summaries[selected_doc])
    
    st.markdown("---")
    
    chat_header_col, export_col = st.columns([5, 1])
    with chat_header_col:
        st.subheader("💬 Active Chat Workspace")
    
    current_history = st.session_state.per_doc_history.get(selected_doc, [])
    if current_history:
        chat_download_string = f"--- CHAT LOG AUDIT FOR FILE: {selected_doc} ---\n\n"
        for msg in current_history:
            chat_download_string += f"[{msg['role'].upper()}]: {msg['content']}\n\n"
        
        with export_col:
            st.download_button(
                label="📥 Export Chat Logs",
                data=chat_download_string,
                file_name=f"Chat_Log_{selected_doc.split('.')[0]}.txt",
                mime="text/plain"
            )

    for message in current_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_query := st.chat_input(f"Ask anything about '{selected_doc}'..."):
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.per_doc_history[selected_doc].append({"role": "user", "content": user_query})

        with st.spinner("Analyzing semantic models and operational blocks..."):
            current_vdb = st.session_state.vector_stores[selected_doc]
            relevant_docs = current_vdb.similarity_search(user_query, k=3)
            doc_context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            web_context = ""
            if enable_web:
                with st.spinner("Executing live multi-node web search routing..."):
                    web_context = search_live_web(user_query)
            
            history_context = ""
            for msg in st.session_state.per_doc_history[selected_doc][-5:]:
                history_context += f"{msg['role'].capitalize()}: {msg['content']}\n"

            full_prompt = (
                f"You are an expert enterprise AI data assistant auditing a document file named '{selected_doc}'.\n"
                f"Use the verified document context, live web search data, and conversation history below to synthesize a highly precise response.\n\n"
                f"Document Database Context:\n{doc_context}\n\n"
                f"Live Web Real-Time Search Context (If Any):\n{web_context}\n\n"
                f"Conversation History:\n{history_context}\n"
                f"User Question: {user_query}\n"
                f"Verified Response:"
            )
            
            response = invoke_groq_llm(full_prompt)
            with st.chat_message("assistant"):
                st.write(response)
            
            st.session_state.per_doc_history[selected_doc].append({"role": "assistant", "content": response})
            st.rerun()

st.markdown("<br><br>", unsafe_allow_html=True)

# ================= BOTTOM SECTION =================
st.markdown("---")
st.subheader("⚙️ System Architecture & Execution Workflow")
st.markdown("""
This engine operates as a secure, local **Retrieval-Augmented Generation (RAG)** computing stack. 
The internal data stream relies on the following process protocols:
* **Ingestion Router Layer:** Automatically captures file streams and runs independent text extractions for `.pdf`, `.docx`, and `.txt` architectures.
* **Semantic Vector Storage:** Chunks data streams using character limit overlaps, creates arrays via local embedding layers, and hosts tables natively inside **ChromaDB**.
* **Cloud LLM Synthesis Integration:** Fires explicit secure tokens directly into the **Groq Cloud Inference Engine** node via zero-dependency `urllib` REST endpoints, leveraging live semantic blocks alongside conversational memory.
""")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(f"""
<div class="dev-profile-card">
    <h3 style="margin-top:0px; border-bottom: 1px solid #282c37; padding-bottom: 8px;">👨‍💻 Engineering Profile & Credentials</h3>
    <p style="margin-top: 12px; margin-bottom:8px;"><b>Lead Architect:</b> {DEV_NAME}</p>
    <p style="margin-bottom:8px;"><b>Institution:</b> {UNIVERSITY}</p>
    <p style="margin-bottom:8px;"><b>Department Core:</b> {DEPARTMENT}</p>
    <p style="margin-bottom:8px;"><b>Contact Portal:</b> {DEV_PHONE}</p>
    <p style="margin-bottom:0px;"><b>Secure Inbox:</b> {DEV_EMAIL}</p>
</div>
""", unsafe_allow_html=True)
