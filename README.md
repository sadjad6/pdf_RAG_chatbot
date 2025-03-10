# DeepSeek-R1 PDF Q&A Assistant

## Overview
The **DeepSeek-R1 PDF Q&A Assistant** is a Streamlit-based web application that enables users to upload a PDF document and ask questions about its content. The application uses **DeepSeek-R1 (1.5B)**, an advanced language model, along with **HuggingFace embeddings** to index and retrieve relevant sections of the PDF for accurate responses.

## Features
- 📄 **Upload and Index PDFs**: Automatically processes and indexes the uploaded document.
- 🤖 **AI-Powered Q&A**: Answers user queries based on the document content using DeepSeek-R1.
- 🔍 **Efficient Retrieval**: Uses **vector search** to find relevant sections of the document.
- 📜 **PDF Preview**: Displays the uploaded document for reference.
- 🎛 **Reset Chat**: Clear previous interactions for a fresh start.
- ⚡ **Fast Response Generation**: Uses **streaming responses** for real-time answers.

## Installation
### Prerequisites
- Python 3.9+
- Ollama installed and running
- 4GB+ free RAM

1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull deepseek-r1:1.5b

### Steps to Install & Run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/sadjad6/pdf_RAG_chatbot.git
   cd pdf_RAG_chatbot
      ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**:
   ```bash
   streamlit run pdf_qa_v2.py
   ```


## How It Works
1. **Upload a PDF** via the sidebar.
2. **The document is indexed** using HuggingFace embeddings and stored for retrieval.
3. **Ask questions** in the chat input field.
4. **DeepSeek-R1** retrieves the most relevant document sections and generates a response.
5. **Streaming response** is displayed in real-time.

## Usage
- Click **"Choose a PDF file"** in the sidebar and upload a document.
- Wait for indexing to complete.
- Enter your question in the chat input box.
- View the response generated by DeepSeek-R1.
- Click **"Clear ↺"** to reset the chat session.

## Version History
- **v4**: Added support for non-textual elements (figures, tables, charts), improved paper search, and enhanced error handling
- **v3**: Added academic paper search functionality
- **v2(Current)**: Improved PDF processing and context retrieval
- **v1**: Basic PDF question answering system

## Contributing
Feel free to fork the repository and submit pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
🚀 **Enjoy exploring your PDFs with AI!**

