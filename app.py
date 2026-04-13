
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import os

groq_api_key = "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"

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Dcoumnet AI Assistant", layout="wide")

# -----------------------------
# CUSTOM CSS (🔥 CORE DESIGN)
# -----------------------------
st.markdown("""
<style>

/* Streamlit background fix */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}

/* Remove white header */
[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    background: transparent;
}

/* Center main container */
.block-container {
    max-width: 900px;
    margin: auto;
}

/* Glass effect */
.glass {
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
}

/* Chat bubbles */
.user-msg {
    background-color: #6366f1;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 10px 0;
    width: fit-content;
    margin-left: auto;
}

.ai-msg {
    background-color: #f1f5f9;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 10px 0;
    width: fit-content;
}

/* Title */
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #4f46e5;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR (LIKE CORTEX)
# -----------------------------
with st.sidebar:
    st.markdown("## 📄 Document Assistant")
    st.markdown("Upload a PDF and ask questions from it.")

    st.markdown("---")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if st.button("Clear Chat"):
        st.session_state.messages = []

    st.markdown("---")
    st.caption("Powered by LLM & Retrieval-Augmented Generation (RAG)")

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="title">Document AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a PDF & ask anything</div>', unsafe_allow_html=True)

# -----------------------------
# CACHE PDF
# -----------------------------
@st.cache_resource
def process_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    return db

# -----------------------------
# CHAT MEMORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# MAIN UI CONTAINER
# -----------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)

if uploaded_file:

    if not groq_api_key:
        st.error("API key missing")
        st.stop()


    db = process_pdf(uploaded_file)
    client = Groq(api_key=groq_api_key)

    # Show chat
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-msg">{msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Ask anything...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        results = db.similarity_search(user_input, k=5)
        context = "\n".join([doc.page_content for doc in results])

        prompt = f"""
You are an intelligent assistant.

STRICT RULES:
- Answer ONLY from the given context
- DO NOT modify names, numbers, or facts
- If exact answer not found, say: "Not found in document"

Context:
{context}

Question:
{user_input}
"""

        with st.spinner(" Thinking..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )

        answer = response.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.rerun()

else:
    st.info("Upload PDF")

st.markdown('</div>', unsafe_allow_html=True)

