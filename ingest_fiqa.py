import os
import json
import sys
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_classic.retrievers import ParentDocumentRetriever
from package import docstore_metadata

# CONFIG
CORPUS_PATH = "fiqa_data/corpus.jsonl"  # Aapke SciFact corpus ka path
COLLECTION_NAME = "FIQA_RAG_DB" # Alag collection name test ke liye
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Initialize Embeddings & Storage
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url=OLLAMA_URL)
docstore = docstore_metadata.docstore

def load_scifact_corpus(filepath):
    print(f"Loading SciFact Corpus from {filepath}...")
    docs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # AllenAI uses 'doc_id', fall back to '_id' just in case
            doc_id = str(data.get("doc_id", data.get("_id", "")))
            
            # Handle text: AllenAI uses 'abstract' (list of strings), BEIR uses 'text' (string)
            title = data.get("title", "")
            if "abstract" in data and isinstance(data["abstract"], list):
                text_content = " ".join(data["abstract"])
            else:
                text_content = data.get("text", "")
                
            content = f"{title}\n{text_content}"
            
            # Preserve doc_id in metadata
            metadata = {
                "doc_id": doc_id,
                "source": "fiqa",
                "vendor_name": "fiqa" 
            }
            docs.append(Document(page_content=content, metadata=metadata))
            
    print(f"Loaded {len(docs)} documents.")
    return docs

raw_docs = load_scifact_corpus(CORPUS_PATH)

# Setup Qdrant
client = QdrantClient(url=QDRANT_URL)
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

# Setup Splitters
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=27)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

print("Starting ingestion into Qdrant & Docstore...")
retriever.add_documents(raw_docs)
print("✅ Ingestion Complete!")