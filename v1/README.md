# 📄 AI PDF Q&A Assistant

## 🔍 Overview
The **AI PDF Q&A Assistant** is a **Streamlit-based web application** that allows users to **upload a PDF** and ask questions about its content using **retrieval-augmented generation (RAG)**. The application leverages **LangChain, Ollama, and an In-Memory Vector Store** for document processing, vector storage, and answering queries efficiently.

## ✨ Features
- 📂 **Upload PDFs** for automatic processing.
- 🔍 **Semantic text chunking** for better document understanding.
- ⚡ **Fast & efficient search** using an in-memory vector store.
- 🤖 **Ollama-powered LLMs** for answering queries.
- 📜 **Inline PDF preview** for easy reference.

---

## 🚀 Installation & Setup
### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/your-repo/pdf-qa-bot.git
cd pdf-qa-bot
```

### 2️⃣ **Install Dependencies**
Create a virtual environment (optional but recommended):
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```
Install the required Python packages:
```sh
pip install -r requirements.txt
```

### 3️⃣ **Run the App**
```sh
streamlit run app.py
```

---

## 📦 Dependencies
Ensure the following packages are installed (included in `requirements.txt`):
```sh
streamlit
langchain
langchain-community
langchain-text-splitters
langchain-core
langchain-ollama
pdfplumber
ollama
```

---

## 🛠 How It Works
### 🔹 **Step 1: Upload a PDF**
- Use the sidebar to upload a PDF.
- The app extracts text using `pdfplumber`.

### 🔹 **Step 2: Text Processing & Indexing**
- The document is split into chunks using `RecursiveCharacterTextSplitter`.
- The vector representations of chunks are stored in an **in-memory vector store**.

### 🔹 **Step 3: Ask Questions**
- Users can type a question in the chat input.
- The app retrieves relevant chunks using similarity search.
- The answer is generated using `deepseek-r1:1.5b` via Ollama.

---

## 📌 File Structure
```plaintext
📂 pdf-qa-bot
├── 📂 pdfs/                # Directory for uploaded PDFs
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── README.md             # Documentation (this file)
└── .gitignore            # Ignore unnecessary files
```

---

## 🖥️ Usage Example
1. Upload a **research paper, article, or any PDF**.
2. Wait for the **processing** to complete.
3. Ask questions about the content (e.g., "What is the main conclusion of this paper?").
4. Get a **concise and relevant response**!

---

## ⚡ Performance Optimization
- **Optimized Prompting:** Uses a structured prompt for efficient responses.
- **Efficient Storage:** Uses an in-memory vector store for fast lookups.
- **Parallel Processing:** Ensures quick PDF text extraction and indexing.

---

## 💡 Future Improvements
✅ Support for **multiple PDFs** at once  
✅ Add **OCR support** for scanned PDFs  
✅ Improve UI with **chat history & citations**  
✅ Deploy the app on **Cloud or Hugging Face Spaces**

---

## 🤝 Contributing
Feel free to fork the repo, submit PRs, or raise issues! Contributions are always welcome. 😊

---

## 📜 License
This project is licensed under the **MIT License**.

