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
import numpy as np
import pytesseract
from sklearn.cluster import DBSCAN
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import ImageNode, TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.multi_modal_llms.ollama import OllamaMultiModal
import streamlit as st

# Initialize session state variables
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.selected_paper = None
    st.session_state.active_query_engine = None
    st.session_state.processed_papers = {}
    st.session_state.extracted_figures = {}
    st.session_state.extracted_tables = {}
    st.session_state.extracted_charts = {}

session_id = st.session_state.id

@st.cache_resource
def load_llm():
    return Ollama(
        model="deepseek-r1:1.5b",
        request_timeout=300.0,
        temperature=0.3,
        base_url='http://localhost:11434'
    )

@st.cache_resource
def load_multimodal_llm():
    """Load a multimodal LLM that can understand images"""
    try:
        return OllamaMultiModal(
            model="llava:7b", 
            base_url='http://localhost:11434',
            request_timeout=300.0,
            temperature=0.3
        )
    except Exception as e:
        st.error(f"Error connecting to Ollama server for multimodal LLM: {str(e)}")
        st.error("Please make sure Ollama is running (http://localhost:11434) and the llava:7b model is installed.")
        st.info("You can install Ollama from https://ollama.com and run 'ollama pull llava:7b' to download the model.")
        return None

def get_image_description(multimodal_llm, image, prompt):
    """Safely get description for an image, handling API changes and errors"""
    try:
        # Convert PIL Image to bytes first
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        # Create ImageDocument
        from llama_index.core.schema import ImageDocument
        image_document = ImageDocument(
            image_path=None,
            image=image_bytes
        )
        
        # Get response as direct string to avoid any API issues
        try:
            response = multimodal_llm.complete(
                prompt=prompt,
                image_documents=[image_document]
            )
            
            # Try different ways to get the string content
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'response'):
                return response.response
            elif hasattr(response, 'raw'):
                return response.raw
            elif isinstance(response, str):
                return response
            else:
                # Last resort: direct string conversion
                return str(response)
        except AttributeError:
            # If there's an attribute error, try to directly convert to string
            return str(response)
    except Exception as e:
        return f"Description not available: {str(e)}"

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

def extract_images(pdf_document):
    """Extract images from PDF document with their positions and page numbers"""
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Get image extension
            ext = base_image["ext"]
            
            try:
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                
                # Get the image rect on page 
                # (fallback to empty if not available, we'll use clustering to identify figures later)
                image_rect = page.get_image_rects(xref)[0] if page.get_image_rects(xref) else fitz.Rect(0, 0, 0, 0)
                
                # Get surrounding text (captions)
                caption = ""
                if image_rect.is_valid and not image_rect.is_empty:
                    # Looking for caption below the image
                    caption_rect = fitz.Rect(image_rect.x0, image_rect.y1, image_rect.x1, min(image_rect.y1 + 150, page.rect.y1))
                    caption = page.get_text("text", clip=caption_rect)
                    # Look for figure reference before the image
                    pre_rect = fitz.Rect(image_rect.x0, max(0, image_rect.y0 - 50), image_rect.x1, image_rect.y0)
                    pre_text = page.get_text("text", clip=pre_rect)
                    if pre_text and "Figure" in pre_text or "Fig." in pre_text:
                        caption = pre_text + "\n" + caption
                
                images.append({
                    "image": image,
                    "page_num": page_num + 1,
                    "position": [image_rect.x0, image_rect.y0, image_rect.x1, image_rect.y1],
                    "caption": caption.strip(),
                    "image_type": "figure" if "Figure" in caption or "Fig." in caption else "image",
                    "file_type": ext
                })
            except Exception as e:
                st.warning(f"Could not extract image: {str(e)}")
    
    # Try to identify figures using clustering if needed
    identify_figures_via_captions(images)
    
    return images

def identify_figures_via_captions(images):
    """Identify figures based on caption texts and improve classification"""
    for i, img in enumerate(images):
        caption = img["caption"].lower()
        
        # Check for figure indicators
        if "figure" in caption or "fig." in caption or "fig " in caption:
            images[i]["image_type"] = "figure"
        
        # Check for table indicators
        elif "table" in caption or "tab." in caption or "tab " in caption:
            images[i]["image_type"] = "table"
        
        # Check for chart/graph indicators
        elif any(word in caption for word in ["chart", "graph", "plot", "histogram", "distribution", "curve"]):
            images[i]["image_type"] = "chart"

def extract_tables(pdf_document):
    """Extract tables from PDF documents"""
    tables = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        # Simple heuristic: tables often have horizontal lines
        # Get all drawings on page
        paths = page.get_drawings()
        
        # Find potential horizontal lines that could be part of tables
        horizontal_lines = []
        for path in paths:
            for item in path["items"]:
                if item[0] == "l":  # Line
                    try:
                        # Handle potential format differences in PyMuPDF versions
                        if isinstance(item[1], (list, tuple)) and len(item[1]) == 4:
                            x0, y0, x1, y1 = item[1]
                        elif isinstance(item[1], dict) and all(k in item[1] for k in ['x0', 'y0', 'x1', 'y1']):
                            # Handle dictionary format
                            x0, y0, x1, y1 = item[1]['x0'], item[1]['y0'], item[1]['x1'], item[1]['y1']
                        else:
                            # Skip if we can't get proper coordinates
                            continue
                            
                        # If line is approximately horizontal
                        if abs(y1 - y0) < 2 and abs(x1 - x0) > 50:
                            horizontal_lines.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
                    except Exception as e:
                        st.warning(f"Error processing line in table extraction: {str(e)}")
                        continue
        
        # Continue with the rest of the function if we found any horizontal lines
        if horizontal_lines:
            # Convert to numpy array for clustering
            lines_array = np.array(horizontal_lines)
            y_positions = lines_array[:, 1]  # Y-coordinates
            
            # Apply clustering to find groups of horizontal lines
            clustering = DBSCAN(eps=50, min_samples=2).fit(y_positions.reshape(-1, 1))
            labels = clustering.labels_
            
            # Process each cluster (potential table)
            for label in set(labels):
                if label == -1:  # Skip noise
                    continue
                    
                # Get lines in this cluster
                cluster_indices = np.where(labels == label)[0]
                cluster_lines = lines_array[cluster_indices]
                
                # Define table boundaries
                min_x = np.min(cluster_lines[:, 0])
                max_x = np.max(cluster_lines[:, 2])
                min_y = np.min(cluster_lines[:, 1]) - 20  # Add margin
                max_y = np.max(cluster_lines[:, 3]) + 20  # Add margin
                
                # Extract table region
                table_rect = fitz.Rect(min_x, min_y, max_x, max_y)
                table_text = page.get_text("text", clip=table_rect)
                
                # Check surrounding text for captions
                caption_rect_above = fitz.Rect(min_x, max(0, min_y - 100), max_x, min_y)
                caption_above = page.get_text("text", clip=caption_rect_above)
                
                caption_rect_below = fitz.Rect(min_x, max_y, max_x, min(max_y + 100, page.rect.y1))
                caption_below = page.get_text("text", clip=caption_rect_below)
                
                # Identify table caption
                caption = ""
                if "Table" in caption_above or "Tab." in caption_above:
                    caption = caption_above
                elif "Table" in caption_below or "Tab." in caption_below:
                    caption = caption_below
                
                # Create table image for visual reference
                table_pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=table_rect)
                table_image = Image.open(io.BytesIO(table_pixmap.tobytes("png")))
                
                tables.append({
                    "content": table_text,
                    "image": table_image,
                    "page_num": page_num + 1,
                    "position": [min_x, min_y, max_x, max_y],
                    "caption": caption.strip()
                })
    
    return tables

def process_pdf(file_path, file_key):
    if file_key not in st.session_state.file_cache:
        node_parser = SentenceSplitter(chunk_size=768, chunk_overlap=150, include_metadata=True)
        llm = load_llm()
        multimodal_llm = load_multimodal_llm()
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", trust_remote_code=True)
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser
        
        with st.spinner("Processing PDF..."):
            # Load the document initially
            loader = SimpleDirectoryReader(input_files=[file_path], filename_as_id=True)
            docs = loader.load_data()
            
            # Open PDF to extract non-textual elements
            pdf_document = fitz.open(file_path)
            
            # Extract images, figures, tables, and charts
            with st.spinner("Extracting figures and tables..."):
                images = extract_images(pdf_document)
                tables = extract_tables(pdf_document)
                
                # Store extracted elements in session state
                st.session_state.extracted_figures[file_key] = [img for img in images if img["image_type"] == "figure"]
                st.session_state.extracted_tables[file_key] = tables
                st.session_state.extracted_charts[file_key] = [img for img in images if img["image_type"] == "chart"]
                
                # Create nodes for textual content
                text_nodes = []
                for doc in docs:
                    text_chunks = node_parser.get_nodes_from_documents([doc])
                    text_nodes.extend(text_chunks)
                
                # Create nodes for figures
                figure_nodes = []
                for i, fig in enumerate(st.session_state.extracted_figures[file_key]):
                    # Analyze figure with multimodal LLM
                    figure_description = ""
                    try:
                        prompt = "Describe this figure and explain what it shows. Be detailed but concise."
                        # Use our helper function instead of direct API calls
                        figure_description = get_image_description(multimodal_llm, fig["image"], prompt)
                    except Exception as e:
                        figure_description = "Figure description not available."
                        st.warning(f"Could not generate figure description: {str(e)}")
                    
                    # Create ImageNode with metadata
                    node_id = f"figure_{file_key}_{i}"
                    metadata = {
                        "page_num": fig["page_num"],
                        "position": fig["position"],
                        "caption": fig["caption"],
                        "description": figure_description,
                        "file_path": file_path,
                        "element_type": "figure"
                    }
                    
                    # Convert bytes to base64 string for ImageNode
                    import base64
                    img_byte_arr = io.BytesIO()
                    fig["image"].save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    # Create image node with base64 string instead of bytes
                    image_node = ImageNode(
                        image=img_base64,
                        image_id=node_id,
                        metadata=metadata,
                        text=f"Figure {i+1}: {fig['caption']}\n{figure_description}"
                    )
                    figure_nodes.append(image_node)
                
                # Create nodes for tables
                table_nodes = []
                for i, table in enumerate(st.session_state.extracted_tables[file_key]):
                    img_byte_arr = io.BytesIO()
                    table["image"].save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Convert bytes to base64 string
                    import base64
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    # Create TextNode for table text content
                    table_text_node = TextNode(
                        text=f"Table {i+1}: {table['caption']}\n{table['content']}",
                        metadata={
                            "page_num": table["page_num"],
                            "position": table["position"],
                            "caption": table["caption"],
                            "file_path": file_path,
                            "element_type": "table"
                        }
                    )
                    
                    # Also create an ImageNode for visual table representation with base64 string
                    table_img_node = ImageNode(
                        image=img_base64,
                        image_id=f"table_img_{file_key}_{i}",
                        metadata={
                            "page_num": table["page_num"],
                            "position": table["position"],
                            "caption": table["caption"],
                            "file_path": file_path,
                            "element_type": "table_image"
                        }
                    )
                    
                    # Connect nodes with relationship
                    table_text_node.relationships[NodeRelationship.VISUAL] = RelatedNodeInfo(
                        node_id=f"table_img_{file_key}_{i}"
                    )
                    
                    table_nodes.append(table_text_node)
                    table_nodes.append(table_img_node)
                
                # Create nodes for charts
                chart_nodes = []
                for i, chart in enumerate(st.session_state.extracted_charts[file_key]):
                    img_byte_arr = io.BytesIO()
                    chart["image"].save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Convert bytes to base64 string
                    import base64
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    # Analyze chart with multimodal LLM
                    chart_description = ""
                    try:
                        prompt = "Analyze this chart/graph and explain what data it shows, trends, patterns, and key insights. Be detailed but concise."
                        # Use our helper function instead of direct API calls
                        chart_description = get_image_description(multimodal_llm, chart["image"], prompt)
                    except Exception as e:
                        chart_description = "Chart description not available."
                        st.warning(f"Could not generate chart description: {str(e)}")
                    
                    # Create ImageNode with metadata
                    node_id = f"chart_{file_key}_{i}"
                    metadata = {
                        "page_num": chart["page_num"],
                        "position": chart["position"],
                        "caption": chart["caption"],
                        "description": chart_description,
                        "file_path": file_path,
                        "element_type": "chart"
                    }
                    
                    # Create image node with base64 string
                    chart_node = ImageNode(
                        image=img_base64,
                        image_id=node_id,
                        metadata=metadata,
                        text=f"Chart {i+1}: {chart['caption']}\n{chart_description}"
                    )
                    chart_nodes.append(chart_node)
                
                # Combine all nodes
                all_nodes = text_nodes + figure_nodes + table_nodes + chart_nodes
        
        with st.spinner("Building index..."):
            # Create index from all nodes
            index = VectorStoreIndex(all_nodes, show_progress=True)
            
            # Create enhanced query engine with visual context
            query_engine = index.as_query_engine(
                streaming=True, 
                similarity_top_k=5,
                response_timeout=60,
                verbose=False
            )
            
            # Update the prompt template to handle visual elements
            qa_prompt_tmpl_str = """Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the question with detailed information from the document including relevant references to figures, tables and charts if they're mentioned in the context.
If the query is about a figure, table, or chart, include specific details about it in your response. 
For figures, describe what they show. For tables, summarize the key data. For charts, explain what trends they illustrate.
Always cite your source (e.g., "According to Figure 3..." or "As shown in Table 2...").

Question: {query_str}
Answer:"""
            
            qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
            query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
            st.session_state.file_cache[file_key] = query_engine
    
    return st.session_state.file_cache[file_key]

def display_visual_elements(file_key):
    """Display the extracted visual elements in the sidebar for reference"""
    
    if file_key in st.session_state.extracted_figures and st.session_state.extracted_figures[file_key]:
        st.sidebar.subheader("Extracted Figures")
        for i, fig in enumerate(st.session_state.extracted_figures[file_key]):
            with st.sidebar.expander(f"Figure {i+1} (Page {fig['page_num']})"):
                st.image(fig["image"], caption=fig["caption"], use_column_width=True)
    
    if file_key in st.session_state.extracted_tables and st.session_state.extracted_tables[file_key]:
        st.sidebar.subheader("Extracted Tables")
        for i, table in enumerate(st.session_state.extracted_tables[file_key]):
            with st.sidebar.expander(f"Table {i+1} (Page {table['page_num']})"):
                st.image(table["image"], caption=table["caption"], use_column_width=True)
                st.text(table["content"])
    
    if file_key in st.session_state.extracted_charts and st.session_state.extracted_charts[file_key]:
        st.sidebar.subheader("Extracted Charts")
        for i, chart in enumerate(st.session_state.extracted_charts[file_key]):
            with st.sidebar.expander(f"Chart {i+1} (Page {chart['page_num']})"):
                st.image(chart["image"], caption=chart["caption"], use_column_width=True)

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
                
                # Display extracted visual elements
                show_visual_elements = st.checkbox("Show Extracted Visual Elements")
                if show_visual_elements:
                    display_visual_elements(file_key)
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
                        
                        # Display extracted visual elements
                        show_visual_elements = st.checkbox("Show Extracted Visual Elements")
                        if show_visual_elements:
                            display_visual_elements(file_key)
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
                        
                        # Display extracted visual elements if available
                        if "active_query_engine" in st.session_state:
                            file_key = f"{session_id}-{st.session_state.selected_paper['title'].replace(' ', '_')}"
                            show_visual_elements = st.checkbox("Show Extracted Visual Elements")
                            if show_visual_elements:
                                display_visual_elements(file_key)
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