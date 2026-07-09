# 📊 Universal Document Intelligence Engine
### Enterprise Hybrid RAG System with Real-Time Web Search & Privacy Controls

An advanced, enterprise-grade **Retrieval-Augmented Generation (RAG)** computing stack built to deliver secure, local document intelligence. The system securely processes local files while leveraging live semantic blocks and real-time web verification.

---

## 🚀 Key Features

* **Ingestion Router Layer:** Automatically captures file streams and runs independent text extractions for `.pdf`, `.docx`, and `.txt` architectures.
* **Semantic Vector Storage:** Chunks data streams using character limit overlaps, creates arrays via local embedding layers, and hosts tables natively inside **ChromaDB**.
* **Isolated LLM Synthesis:** Fires explicit prompts inside a local **Ollama Llama 3** node, leveraging conversational memory and vector context[cite: 7].
* **Hybrid Intelligence Layer:** Features automated live web search routing via DuckDuckGo API to dynamically expand context for public facts[cite: 7].
* **🛡️ Privacy & Compliance Module:** Includes regular expression-based **PII Data Masking** to safely redact sensitive user data (emails and contact numbers) before processing[cite: 7].
* **Premium Dashboard Layout:** Built using **Streamlit** with an executive summary panel, automated metadata tracking, and chat log auditing tools[cite: 7].

---

## 🛠️ Tech Stack

* **Frontend/Workspace:** Streamlit[cite: 7]
* **Orchestration:** LangChain / LangChain Community[cite: 7, 8]
* **LLM Engine:** Ollama (Llama 3 Local Node)[cite: 7]
* **Vector Database:** ChromaDB[cite: 7, 8]
* **Embeddings:** HuggingFace Embeddings (`all-MiniLM-L6-v2`)[cite: 7]
* **Data Parsers:** PyPDFLoader, docx2txt[cite: 7, 8]

---

## ⚙️ Project Architecture & Execution Workflow

1. **Upload & Extract:** User uploads a file through the Streamlit Sidebar Control Center[cite: 7].
2. **PII Masking:** If enabled, the system automatically sanitizes sensitive fields like emails and phone numbers[cite: 7].
3. **Vector Database Ingestion:** Documents are split into overlapping chunks, vectorized, and stored in ChromaDB[cite: 7].
4. **Hybrid Context Generation:** When a query is made, the engine pulls the top 3 matching document chunks and blends them with real-time web search results (if active)[cite: 7].
5. **LLM Execution:** The combined context and full conversation history are formatted into a precise prompt and synthesized via the local Llama 3 instance[cite: 7].

---

## 💻 How to Setup and Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/abhargav123/universal-document-intelligence.git](https://github.com/abhargav123/universal-document-intelligence.git)
cd universal-document-intelligence
### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
ollama pull llama3
streamlit run app.py
