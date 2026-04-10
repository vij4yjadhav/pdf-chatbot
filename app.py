
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="PDF Chatbot", layout="centered")

# 🔝 VERY TOP (after imports)
st.markdown("""
<h1 style='text-align: center;'>🤖 PDF Chatbot</h1>
<p style='text-align: center;'>Ask anything from your document</p>
""", unsafe_allow_html=True)
st.caption("Chat with your PDF like ChatGPT")

# -----------------------------
# Cache PDF Processing
# -----------------------------
@st.cache_resource
def process_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    return db

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []

# -----------------------------
# Chat History (IMPORTANT)
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Process PDF
# -----------------------------
if uploaded_file and groq_api_key:

    db = process_pdf(uploaded_file)
    client = Groq(api_key=groq_api_key)

    st.success("✅ PDF ready! Start chatting 👇")

    # Show old messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask something about your PDF...")

    if user_input:
        # Show user message
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Retrieve context
        results = db.similarity_search(user_input, k=5)
        context = "\n".join([doc.page_content for doc in results])

        prompt = f"""
You are an intelligent assistant.

Answer ONLY from the given context.
If answer is not present, say "Not found in document".

Context:
{context}

Question:
{user_input}
"""
        with st.spinner("🤖 Thinking..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}])
        

        answer = response.choices[0].message.content

        # Show AI message
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Please upload a PDF and enter API key from the sidebar")