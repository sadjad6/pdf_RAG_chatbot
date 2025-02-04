import os
import fitz
import base64
import io
from PIL import Image
import gc
import tempfile
import uuid
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

@st.cache_resource
def load_llm():
    # Increased timeout and added temperature parameter
    return Ollama(
        model="deepseek-r1:1.5b",
        request_timeout=300.0,  
        temperature=0.3,        # For more focused responses
        base_url='http://localhost:11434'  # Explicit Ollama server URL
    )

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    """Display all pages of the PDF as images."""
    st.markdown("### PDF Preview")

    # Read the PDF into PyMuPDF
    file.seek(0)
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")

    images = []

    for page_num in range(len(pdf_document)):  # Process all pages
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # High resolution
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)

    # Display extracted pages as images
    for img in images:
        st.image(img, use_container_width=True)  


with st.sidebar:
    st.header("Upload Your Document")
    uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):
                    # Optimized processing parameters
                    node_parser = SentenceSplitter(
                        chunk_size=768,  
                        chunk_overlap=150,
                        include_metadata=False  
                    )
                    
                    llm = load_llm()
                    embed_model = HuggingFaceEmbedding(
                        model_name="BAAI/bge-base-en-v1.5",  
                        trust_remote_code=True
                    )

                    Settings.llm = llm
                    Settings.embed_model = embed_model
                    Settings.node_parser = node_parser

                    # Load documents with progress
                    with st.spinner("Processing PDF pages..."):
                        loader = SimpleDirectoryReader(
                            input_dir=temp_dir,
                            required_exts=[".pdf"],
                            filename_as_id=True  # document tracking
                        )
                        docs = loader.load_data()

                    # Create index with optimized settings
                    with st.spinner("Building search index..."):
                        index = VectorStoreIndex.from_documents(
                            docs, 
                            show_progress=True,
                            use_async=True  # Enable async processing
                        )
                    
                    # Configure query engine with timeout
                    query_engine = index.as_query_engine(
                        streaming=True,
                        similarity_top_k=3,  
                        response_timeout=60,  # Query-specific timeout
                        verbose=False         # Disable debug output
                    )

                    # Streamlined prompt template
                    qa_prompt_tmpl_str = (
                        "Context:\n{context_str}\n\n"
                        "Question: {query_str}\n"
                        "Answer concisely using only the context: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                    query_engine.update_prompts({
                        "response_synthesizer:text_qa_template": qa_prompt_tmpl
                    })
                    
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            st.stop()

st.header("DeepSeek-R1 PDF Q&A Assistant")
st.button("Clear ↺", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("Analyzing document..."):  # Add progress indicator
                streaming_response = query_engine.query(prompt)
                
                if streaming_response.response_gen:
                    for chunk in streaming_response.response_gen:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                else:
                    full_response = "I couldn't find relevant information in the document."
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Please try rephrasing your question or check the document content."})
