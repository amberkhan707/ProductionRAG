# Agentic RAG API

A Multi-Vendor, Metadata-Aware Retrieval Augmented Generation System.

This project implements an advanced Agentic RAG architecture designed to parse complex PDF documentation, intelligently extract metadata (vendors, sections), and perform hybrid retrieval with reranking. It utilizes LangGraph for workflow orchestration, Qdrant for vector storage, and Docling for high-fidelity document parsing with OCR.

## 🚀 Features

- **Intelligent Ingestion & OCR**: Uses Docling to process PDFs, applying "Sticky Header" logic to preserve context (headers/sections) across page boundaries.
- **Hybrid Retrieval**: Combines Dense Vector Search (Qdrant) and Sparse Search (BM25) to maximize retrieval recall.
- **Agentic Query Analysis**: Dynamically extracts metadata filters (Vendor, Section) from natural language queries using LLMs.
- **Relevance Grading**: A dedicated graph node grades retrieved documents for relevance before generation to reduce hallucinations.
- **Cross-Encoder Reranking**: Uses HuggingFace Cross-Encoders to re-order retrieved context for maximum precision.
- **Parent-Child Indexing**: Retrieves full parent documents based on matching smaller child chunks to maintain context window integrity.

## 🛠 Tech Stack

- **Orchestration**: LangChain & LangGraph
- **API Framework**: FastAPI
- **Vector Database**: Qdrant
- **LLM & Embeddings**:
  - Inference: Groq (Llama 3.3 / Llama 4)
  - Embeddings: Ollama (nomic-embed-text)
- **Document Processing**: Docling
- **Reranking**: HuggingFace (BAAI/bge-reranker-v2-m3)

## 📋 Prerequisites

Ensure you have the following installed and running:

- Python 3.10+
- **Qdrant**: Running on port 6333.

  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  
- **Ollama**: Running locally with the embedding model pulled.

  ```bash
  ollama serve
  ollama pull nomic-embed-text
  
Groq API Key: Required for the LLM inference.


## ⚙️ Configuration

Create a `.env` file in the root directory:

```
# API Keys
GROQ_API_KEY=your_groq_api_key_here
HF_API_KEY=your_huggingface_token  # Optional, usually needed for gated models

# Services
QDRANT_URL=http://localhost:6333
OLLAMA_BASE_URL=http://localhost:11434
```
## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/agentic-rag-api.git
cd agentic-rag-api
```
Install dependencies:
```
pip install -r requirements.txt
```
Note: Ensure you have the custom package module available in your python path if it contains shared logic (docstore_metadata, bm25builder, etc.).

## 📖 Usage

### 1. Ingest Documents

Place your PDF files into the `documents/` directory.  
The ingestion pipeline performs OCR, chunking, and indexing into Qdrant.

```bash
python Ingest.py
```
Note: This process initializes the "Parent Document Retriever" structure. It may take time depending on OCR usage and file size.

### 2. Start the API Server

Launch the FastAPI application:

```
python app.py
Server will start at http://0.0.0.0:8000
```

### 3. Query the Agent

You can interact with the API via the endpoint /chat.

Example Request:

```
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the performance metrics for the Solar Turbine model?"}'
```

Example Response:

```json
{
  "answer": "Based on the Solar Turbine documentation, the performance metrics include..."
}
```

## 🏗 Architecture Workflow

The system uses a **LangGraph `StateGraph`** to orchestrate the end-to-end request lifecycle:

1. **Analyze Query**  
   The LLM parses the user question to extract intent, vendor identifiers, and relevant document sections.

2. **Retrieve**
   - Applies metadata-based filtering (Vendor, Section) within Qdrant
   - Executes hybrid retrieval:
     - Dense vector search
     - Sparse BM25 search
   - Reranks retrieved candidates using cross-encoders

3. **Grade Documents**  
   An LLM evaluates retrieved chunks for semantic relevance. Non-relevant context is discarded.

4. **Generate**  
   The final response is generated using only high-quality, validated context.

---

## 📂 Project Structure

```text
├── documents/               # PDF input directory
├── persistent_doc_store/    # Local storage for parent documents
├── package/                 # Shared utilities (BM25, metadata logic)
├── Ingest.py                # ETL pipeline: PDF → OCR → Qdrant
├── app.py                   # FastAPI application and LangGraph workflow
├── .env                     # Environment configuration
└── README.md
```
