import streamlit as st
import urllib.request
import json
import re

# Premium Slate-Dark UI Override
st.set_page_config(page_title="Universal Document Intelligence Engine", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #1E293B; color: #E2E8F0; }
    .sidebar .sidebar-content { background-color: #0F172A; }
    h1, h2, h3 { color: #38BDF8; }
    </style>
""", unsafe_allow_html=True)

# Sidebar Branding
st.sidebar.title("🏢 Enterprise Architecture")
st.sidebar.markdown("### **Lead Developer:**")
st.sidebar.markdown("**Anupam Bhargav**")
st.sidebar.markdown("Computer Science & Engineering\nMMMUT")

st.title("🚀 Universal Document Intelligence Engine")
st.write("Enterprise-grade Hybrid RAG & PII Masking Pipeline")

# PII Masking Component
def mask_pii(text):
    email_regex = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_regex = r'\+?\d{10,12}'
    text = re.sub(email_regex, "[REDACTED_EMAIL]", text)
    text = re.sub(phone_regex, "[REDACTED_PHONE]", text)
    return text

# Core Interface
uploaded_file = st.file_uploader("Upload Document (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
