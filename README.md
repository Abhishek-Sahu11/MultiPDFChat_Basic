# 📚 Multi-PDF RAG Chatbot

A basic Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDF documents and ask questions about their contents using natural language.

Built with **Python, LangChain, Google Gemini, Hugging Face Sentence Transformers, and Streamlit**.

🌐 **Live Demo:** [Multi-PDF RAG Chatbot](https://multipdfchatbasic.streamlit.app/)

📂 **GitHub Repository:** [MultiPDFChat_Basic](https://github.com/Abhishek-Sahu11/MultiPDFChat_Basic)

## ✨ Features

- **Multiple PDF Upload:** Upload multiple PDF documents simultaneously.
- **PDF Text Extraction:** Extract text from uploaded PDFs using PyPDF.
- **Text Chunking:** Split extracted text into smaller, overlapping chunks.
- **Local Embeddings:** Generate text embeddings using the Sentence Transformers model.
- **Vector Search:** Store and retrieve document chunks using LangChain's InMemoryVectorStore.
- **AI-Powered Question Answering:** Generate context-based answers using Google Gemini.
- **Source Tracking:** Display the source PDF filename and page number associated with retrieved content.
- **Chat Interface:** Interact with uploaded documents through a Streamlit chat interface.

## 🏗️ RAG Architecture

The application follows this basic RAG pipeline:

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Document Chunking
    ↓
Embedding Generation
    ↓
In-Memory Vector Store
    ↓
User Question
    ↓
Similarity Search
    ↓
Retrieve Relevant Chunks
    ↓
Construct Prompt with Context
    ↓
Google Gemini
    ↓
Context-Based Answer
    ↓
Display Answer and Sources
```

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Google Gemini |
| LLM Integration | LangChain Google GenAI |
| Embedding Model | Sentence Transformers |
| Embedding Integration | LangChain Hugging Face |
| PDF Processing | PyPDF |
| Text Splitting | RecursiveCharacterTextSplitter |
| Vector Store | InMemoryVectorStore |
| Environment Management | Python dotenv |
| Deployment | Streamlit Community Cloud |

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Abhishek-Sahu11/MultiPDFChat_Basic.git

cd MultiPDFChat_Basic
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Google API key

Obtain a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/).

Create a `.env` file in the project root directory:

```env
GOOGLE_API_KEY="your_google_api_key"
```

**Important:** Never upload your `.env` file or expose your API key publicly.

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 🚀 How to Use

1. Open the application.
2. Upload one or more PDF documents using the sidebar.
3. Click **Process PDFs**.
4. Wait for the documents to be extracted, chunked, and embedded.
5. Enter a question in the chat input.
6. Receive an answer generated using the retrieved document context.
7. Expand **View retrieved sources** to see the relevant PDF filenames and page numbers.

## 🧠 Technical Implementation

### Document Processing
- Extracts text page by page from uploaded PDFs.
- Preserves source filename and page number as document metadata.
- Splits documents using `RecursiveCharacterTextSplitter`.
- Chunk size: 1,000 characters.
- Chunk overlap: 150 characters.

### Embedding Generation
Uses the locally executed Hugging Face model:

`sentence-transformers/all-MiniLM-L6-v2`

The model converts document chunks into numerical vector representations for semantic search.

### Vector Storage and Retrieval
Uses LangChain's `InMemoryVectorStore` to store document embeddings during the application session.

Similarity search retrieves the four most relevant document chunks for a user query.

### Response Generation
The retrieved chunks are incorporated into a prompt and passed to Google Gemini.

The prompt instructs the model to:
- Answer using the provided document context.
- State when the answer cannot be found in the uploaded documents.
- Treat document content as reference material rather than instructions.

## 📁 Project Structure

```text
MultiPDFChat_Basic/
│
├── app.py
├── requirements.txt
├── .gitignore
├── .env                 # Local API key (not committed)
└── .venv/               # Virtual environment (not committed)
```

## 🔐 Security and Data Handling

- API credentials are stored outside the source code.
- Local credentials are maintained in the `.env` file.
- Streamlit Cloud credentials are configured through its Secrets settings.
- Uploaded PDFs and the in-memory vector store are not persisted by this application as a permanent document database.

## ⚠️ Current Limitations

This is a basic learning project demonstrating the core RAG workflow.

- The vector store is in-memory and is not persistent.
- Documents need to be processed again when the session state is lost.
- Chat history is maintained within the Streamlit session, but previous conversation turns are not included in the LLM prompt.
- Scanned PDFs without extractable text require OCR, which is not currently implemented.
- Retrieval quality depends on document content, chunking, embeddings, and similarity search.

## 🎯 Project Objective

The objective of this project is to understand and implement the fundamental components of a Retrieval-Augmented Generation system, including document ingestion, text extraction, chunking, embeddings, vector storage, semantic retrieval, prompt construction, and LLM-based response generation.

This project serves as a foundation for exploring more advanced RAG architectures and production-oriented AI applications.

---

**Developed by Abhishek Sahu**

[GitHub](https://github.com/Abhishek-Sahu11) · [Live Application](https://multipdfchatbasic.streamlit.app/)
