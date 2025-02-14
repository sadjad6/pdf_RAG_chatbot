import os
import fitz
import io
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from PIL import Image
import gc
import uuid
import requests
import re
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st

# Initialize session state variables
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.selected_paper = None
    st.session_state.active_query_engine = None
    st.session_state.processed_papers = {}

session_id = st.session_state.id

@st.cache_resource
def load_llm():
    return Ollama(
        model="deepseek-r1:1.5b",
        request_timeout=300.0,
        temperature=0.3,
        base_url='http://localhost:11434'
    )

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    st.markdown("### PDF Preview")
    file.seek(0)
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    for img in images:
        st.image(img, use_container_width=True)

def display_pdf_from_url(pdf_url):
    try:
        response = requests.get(pdf_url)
        pdf_document = fitz.open(stream=response.content, filetype="pdf")
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
        for img in images:
            st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

def validate_arxiv_url(url):
    """Ensure proper arXiv PDF URL format"""
    if 'arxiv.org' in url:
        if '/abs/' in url:
            url = url.replace('/abs/', '/pdf/') + '.pdf'
        if not url.endswith('.pdf'):
            url += '.pdf'
    return url

def download_pdf_from_url(pdf_url, download_dir, paper_title):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    try:
        pdf_url = validate_arxiv_url(pdf_url)
        response = requests.get(pdf_url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        sanitized_title = re.sub(r'[^\w._-]', '_', paper_title).strip('_')
        file_name = os.path.join(download_dir, f"{sanitized_title}.pdf")
        
        with open(file_name, "wb") as pdf_file:
            for chunk in response.iter_content(chunk_size=8192):
                pdf_file.write(chunk)
        return file_name
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            st.warning(f"Could not download PDF automatically. Please download it manually.")
            #st.markdown(f"[{pdf_url}]({pdf_url})")
            st.info("After downloading, use the 'Upload PDF' option to ask questions about the document.")
            return None
        else:
            st.error(f"Download error ({e.response.status_code}): Could not retrieve PDF from provided URL")
            return None
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return None

def search_semantic_scholar(paper_title):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {'query': paper_title, 'limit': 5, 'fields': 'title,openAccessPdf'}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data.get('data'):
            return None
        for paper in data['data']:
            if paper['title'].lower().strip() == paper_title.lower().strip():
                pdf_info = paper.get('openAccessPdf')
                if pdf_info and pdf_info.get('url'):
                    return pdf_info['url']
        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:  # Rate limit error
            return None  # Silently handle rate limit
        st.error(f"Semantic Scholar error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Semantic Scholar connection error: {str(e)}")
        return None

def search_arxiv_exact(paper_title):
    query = f'ti:"{paper_title}"'
    encoded_query = quote_plus(query)
    url = f'http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results=1'
    try:
        response = requests.get(url)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        entries = root.findall('{http://www.w3.org/2005/Atom}entry')
        if entries:
            title_elem = entries[0].find('{http://www.w3.org/2005/Atom}title')
            if title_elem is not None:
                arxiv_title = title_elem.text.strip()
                if arxiv_title.lower() == paper_title.lower():
                    # Get PDF link from entry
                    for link in entries[0].findall('{http://www.w3.org/2005/Atom}link'):
                        if link.get('title') == 'pdf':
                            return link.get('href')
        return None
    except Exception as e:
        st.error(f"arXiv exact search error: {str(e)}")
        return None

def search_arxiv_similar(paper_title, max_results=5):
    try:
        encoded_query = quote_plus(paper_title)
        url = f'http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}&sortBy=relevance'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        similar_papers = []
        namespace = '{http://www.w3.org/2005/Atom}'
        
        for entry in root.findall(f'{namespace}entry'):
            try:
                title_elem = entry.find(f'{namespace}title')
                if title_elem is None:
                    continue
                
                title = title_elem.text.strip().replace('\n', ' ')
                
                # Find PDF link in entry
                pdf_url = None
                for link in entry.findall(f'{namespace}link'):
                    if link.get('title') == 'pdf':
                        pdf_url = link.get('href')
                        break
                
                if pdf_url:
                    similar_papers.append({'title': title, 'url': pdf_url})
            except (AttributeError, IndexError):
                continue
                
        return similar_papers if similar_papers else None
    except Exception as e:
        st.error(f"arXiv similar search error: {str(e)}")
        return None

def get_any_similar_papers(paper_title):
    papers = search_arxiv_similar(paper_title)
    if papers:
        return papers
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {'query': paper_title, 'limit': 5, 'fields': 'title,openAccessPdf'}
        response = requests.get(url, params=params)
        results = response.json().get('data', [])
        return [{'title': p['title'], 'url': p['openAccessPdf']['url']} for p in results if p.get('openAccessPdf')]
    except Exception:
        return None

def search_pdf_url(paper_title):
    try:
        semantic_url = search_semantic_scholar(paper_title)
        if semantic_url:
            return semantic_url
        arxiv_url = search_arxiv_exact(paper_title)
        return arxiv_url if arxiv_url else None
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None

def process_pdf(file_path, file_key):
    if file_key not in st.session_state.file_cache:
        node_parser = SentenceSplitter(chunk_size=768, chunk_overlap=150, include_metadata=False)
        llm = load_llm()
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", trust_remote_code=True)
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser
        with st.spinner("Processing PDF..."):
            loader = SimpleDirectoryReader(input_files=[file_path], filename_as_id=True)
            docs = loader.load_data()
        with st.spinner("Building index..."):
            index = VectorStoreIndex.from_documents(docs, show_progress=True, use_async=True)
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=3, response_timeout=60, verbose=False)
        qa_prompt_tmpl_str = "Context:\n{context_str}\n\nQuestion: {query_str}\nAnswer concisely using only the context: "
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
        st.session_state.file_cache[file_key] = query_engine
    return st.session_state.file_cache[file_key]

with st.sidebar:
    st.header("Upload or Search Document")
    doc_source = st.radio("Source:", ["Upload PDF", "Search by Title"])
    download_dir = os.path.join(os.getcwd(), "downloaded_papers")
    os.makedirs(download_dir, exist_ok=True)

    if doc_source == "Upload PDF":
        uploaded_file = st.file_uploader("Choose PDF", type="pdf")
        if uploaded_file:
            try:
                file_path = os.path.join(download_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                file_key = f"{session_id}-{uploaded_file.name}"
                query_engine = process_pdf(file_path, file_key)
                st.session_state.active_query_engine = query_engine
                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()
    else:
        paper_title_search = st.text_input("Paper Title:")
        if paper_title_search:
            st.write("Searching...")
            pdf_url = search_pdf_url(paper_title_search)
            if pdf_url:
                st.write(f"Found PDF: {pdf_url}")
                pdf_file_path = download_pdf_from_url(pdf_url, download_dir, paper_title_search)
                if pdf_file_path:
                    file_key = f"{session_id}-{os.path.basename(pdf_file_path)}"
                    try:
                        query_engine = process_pdf(pdf_file_path, file_key)
                        st.session_state.active_query_engine = query_engine
                        st.success("Ready to Chat!")
                        with open(pdf_file_path, 'rb') as f:
                            display_pdf(f)
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
                        st.stop()
                else:
                    st.markdown(f"**Direct PDF Link:** [Download manually]({pdf_url})")
            else:
                st.warning("Exact paper not found. Searching for similar...")
                similar_papers = get_any_similar_papers(paper_title_search)
                if similar_papers:
                    st.subheader("Suggested Related Papers:")
                    for paper in similar_papers[:3]:
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{paper['title']}**")
                        with col2:
                            st.markdown(f"[PDF]({paper['url']})", unsafe_allow_html=True)
                        with col3:
                            if st.button("Select", key=f"select_{paper['url']}"):
                                with st.spinner("Loading selected paper..."):
                                    try:
                                        pdf_path = download_pdf_from_url(paper['url'], download_dir, paper['title'])
                                        if pdf_path:
                                            file_key = f"{session_id}-{os.path.basename(pdf_path)}"
                                            query_engine = process_pdf(pdf_path, file_key)
                                            st.session_state.active_query_engine = query_engine
                                            st.session_state.selected_paper = paper
                                            reset_chat()
                                            st.rerun()
                                        else:
                                            st.error("Failed to download selected paper")
                                    except Exception as e:
                                        st.error(f"Error loading paper: {str(e)}")
                    if st.session_state.selected_paper:
                        st.success(f"Loaded: {st.session_state.selected_paper['title']}")
                        display_pdf_from_url(st.session_state.selected_paper['url'])
                else:
                    st.error("Couldn't find similar papers. Try these alternatives:")
                    st.markdown("""
                    - Search on [arXiv.org](https://arxiv.org) directly
                    - Try [Google Scholar](https://scholar.google.com)
                    - Check paper title spelling
                    - Use more specific keywords
                    """)

st.header("DeepSeek-R1 PDF Q&A Assistant")
st.button("Clear Chat", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if 'active_query_engine' in st.session_state:
    if prompt := st.chat_input("Ask about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                streaming_response = st.session_state.active_query_engine.query(prompt)
                if streaming_response.response_gen:
                    for chunk in streaming_response.response_gen:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("No response generated")
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("Upload PDF or search paper title to begin")