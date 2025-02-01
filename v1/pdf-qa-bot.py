import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os
import base64

# ---- App Config ----
st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄", layout="wide")

# ---- Custom Styling ----
st.markdown(
    """
    <style>
        .stChatMessage { border-radius: 10px; padding: 10px; }
        .user-message { background-color: #E3F2FD; }
        .assistant-message { background-color: #E8F5E9; }
        .stButton>button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Header Section ----
st.markdown("<h1 style='text-align: center;'>📄 AI PDF Q&A Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a PDF and ask questions about its content.</p>", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.header("Upload Your PDF")

pdfs_directory = "pdfs/"
if not os.path.exists(pdfs_directory):
    os.makedirs(pdfs_directory)

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# ---- AI Components ----
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="deepseek-r1:1.5b")

template = """
You are an assistant for question-answering tasks. Use the following retrieved context to answer the question. If unsure, just say you don't know. Answer in three concise sentences.
Question: {question} 
Context: {context} 
Answer:
"""

def upload_pdf(file):
    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_documents(documents)

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# ---- PDF Processing & Chat ----
if uploaded_file:
    pdf_path = upload_pdf(uploaded_file)

    # Convert PDF to base64 for inline display
    with open(pdf_path, "rb") as pdf_file:
        pdf_base64 = base64.b64encode(pdf_file.read()).decode("utf-8")

    # Display PDF in Sidebar
    st.sidebar.success("✅ PDF Uploaded Successfully!")
    st.sidebar.subheader("📜 Preview PDF")
    pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="600px"></iframe>'
    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)
    
    with st.spinner("Processing PDF... 📄"):
        documents = load_pdf(pdf_path)
        chunked_documents = split_text(documents)
        index_docs(chunked_documents)
    st.success("✅ Processing Complete! You can now ask questions.")

    # Chat Input after processing
    question = st.chat_input("Ask me anything about the PDF...")

    if question:
        st.chat_message("user").write(question)
        with st.spinner("Thinking... 🤔"):
            related_documents = retrieve_docs(question)
            answer = answer_question(question, related_documents)

        st.chat_message("assistant").write(answer)
