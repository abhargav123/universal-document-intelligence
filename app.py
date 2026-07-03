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

st.set_page_config(page_title="Universal Doc AI Engine", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0c10 !important; color: #ffffff !important; }
    header[data-testid="stHeader"] { background-color: transparent !important; }
    p, span, label, li, div, h1, h2, h3, h4 { color: #ffffff !important; }
    section[data-testid="stSidebar"] { background-color: #14161d !important; border-right: 1px solid #282c37; }
    .dev-profile-card { background-color: #14161d; padding: 25px; border-radius: 8px; border: 1px solid #282c37; margin-top: 15px; margin-bottom: 25px; }
    div[data-testid="stMetricValue"] { color: #00ffd0 !important; font-weight: 700 !important; font-size: 26px !important; }
    div[data-testid="metric-container"] { background-color: #14161d !important; border: 1px solid #282c37 !important; border-radius: 8px; padding: 15px; }
    [data-testid="stFileUploader"] { background-color: #14161d !important; border: 1px solid #282c37 !important; padding: 10px !important; border-radius: 8px !important; }
    [data-testid="stFileUploaderDropzone"] { background-color: #1a1d26 !important; border: 2px dashed #454e6d !important; border-radius: 6px !important; }
    [data-testid="stFileUploaderDropzone"] *, .stFileUploader * { color: #ffffff !important; }
    [data-testid="stFileUploaderDropzone"] button { background-color: #282c37 !important; border: 1px solid #454e6d !important; color: #ffffff !important; }
    .streamlit-expanderHeader { background-color: #14161d !important; border: 1px solid #282c37 !important; color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

DEV_NAME = "Anupam Bhargav"
UNIVERSITY = "Madan Mohan Malaviya University of Technology"
DEPARTMENT = "Computer Science & Engineering Department"
DEV_EMAIL = "abhargav3637@gmail.com"
DEV_PHONE = "+91 8081300874"

st.title("Universal Document Intelligence Engine")
st.caption("Enterprise Hybrid RAG System with Real-Time Web Search & Core Local Document Context.")
st.markdown("---")

if "vector_stores" not in st.session_state: st.session_state.vector_stores = {}
if "doc_summaries" not in st.session_state: st.session_state.doc_summaries = {}
if "per_doc_history" not in st.session_state: st.session_state.per_doc_history = {}
if "doc_stats" not in st.session_state: st.session_state.doc_stats = {}

if not os.path.exists("extracted_pdf"): os.makedirs("extracted_pdf")

@st.cache_resource
def load_system():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_system()

def invoke_groq_llm(prompt):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return "Error: GROQ_API_KEY config variable missing in Streamlit Secrets."
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Enterprise Inference Pipeline Offline: {str(e)}"

def search_live_web(query):
    try:
        enc_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={enc_query}&format=json&no_html=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
        if data.get("AbstractText"): return f"Source: DuckDuckGo - {data['AbstractText']}"
        return "No direct public facts found."
    except: return "Web intelligence bridge temporary offline."

def mask_sensitive_data(text):
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\b\d{10}\b', '[REDACTED_CONTACT]', text)
    return text

def analyze_text_metadata(text_content):
    words = re.findall(r'\b\w+\b', text_content.lower())
    total_words = len(words)
    filtered = [w for w in words if w not in ["the", "and", "for", "you", "this"] and len(w) > 2]
    keywords = [item[0] for item in Counter(filtered).most_common(5)]
    return {"words": total_words, "keywords": keywords, "time": max(1, round(total_words / 200))}

def extract_text_by_format(file_path, file_name, mask_enabled):
    ext = file_name.split(".")[-1].lower()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if mask_enabled:
            for d in docs: d.page_content = mask_sensitive_data(d.page_content)
        return text_splitter.split_documents(docs), " ".join([d.page_content for d in docs])
    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f: text = f.read()
        if mask_enabled: text = mask_sensitive_data(text)
        doc = Document(page_content=text, metadata={"source": file_name})
        return text_splitter.split_documents([doc]), text
    return None, None

st.sidebar.header("📁 Document Control Center")
uploaded_files = st.sidebar.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True)
mask_data = st.sidebar.toggle("Enable PII Data Masking", value=False)

active_docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        active_docs.append(uploaded_file.name)
        file_path = os.path.join("extracted_pdf", uploaded_file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        if uploaded_file.name not in st.session_state.vector_stores:
            chunks, raw_text = extract_text_by_format(file_path, uploaded_file.name, mask_data)
            if chunks:
                st.session_state.vector_stores[uploaded_file.name] = Chroma.from_documents(chunks, embeddings)
                st.session_state.per_doc_history[uploaded_file.name] = []
                st.session_state.doc_stats[uploaded_file.name] = analyze_text_metadata(raw_text)
                summary_prompt = f"Provide a concise 2-sentence summary and 3 key insights for:\n{raw_text[:2500]}"
                st.session_state.doc_summaries[uploaded_file.name] = invoke_groq_llm(summary_prompt)

selected_doc = None
if active_docs:
    selected_doc = st.sidebar.selectbox("Choose document:", active_docs)
    enable_web = st.sidebar.toggle("Enable Live Web Search", value=False)

if selected_doc:
    st.subheader(f"📊 Insights Dashboard: {selected_doc}")
    stats = st.session_state.doc_stats[selected_doc]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Words", f"{stats['words']}")
    col2.metric("Reading Time", f"{stats['time']} min")
    col3.metric("Keywords", ", ".join(stats['keywords']).title())
    
    with st.expander("📝 Document Executive Summary", expanded=True):
        st.markdown(st.session_state.doc_summaries[selected_doc])
    
    st.subheader("💬 Active Chat Workspace")
    current_history = st.session_state.per_doc_history.get(selected_doc, [])
    for message in current_history:
        with st.chat_message(message["role"]): st.write(message["content"])

    if user_query := st.chat_input("Ask anything..."):
        with st.chat_message("user"): st.write(user_query)
        st.session_state.per_doc_history[selected_doc].append({"role": "user", "content": user_query})

        current_vdb = st.session_state.vector_stores[selected_doc]
        relevant_docs = current_vdb.similarity_search(user_query, k=3)
        doc_context = "\n\n".join([d.page_content for d in relevant_docs])
        
        web_context = search_live_web(user_query) if enable_web else ""
        
        full_prompt = (
            f"Context:\n{doc_context}\n\nWeb:\n{web_context}\n\nQuestion: {user_query}\nAnswer:"
        )
        response = invoke_groq_llm(full_prompt)
        with st.chat_message("assistant"): st.write(response)
        st.session_state.per_doc_history[selected_doc].append({"role": "assistant", "content": response})
        st.rerun()

st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown(f"""
<div class="dev-profile-card">
    <h3>👨‍💻 Engineering Profile</h3>
    <p><b>Lead Architect:</b> {DEV_NAME}</p>
    <p><b>Institution:</b> {UNIVERSITY}</p>
    <p><b>Department:</b> {DEPARTMENT}</p>
</div>
""", unsafe_allow_html=True)
