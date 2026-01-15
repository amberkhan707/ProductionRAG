import os
import sys
import pickle
import re
import shutil

# LOADERS
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# TEXT SPLITTERS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# EMBEDDINGS
from langchain_ollama.embeddings import OllamaEmbeddings

# VECTOR STORE (QDRANT)
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# CLASSIC RAG
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, EncoderBackedStore

# CONFIG
DOC_DIR = "documents"
DOC_STORE_PATH = "./persistent_doc_store"
COLLECTION_NAME = "agentic_rag_db"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- OCR Configuration ---
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = EasyOcrOptions()
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# PRECHECKS
if not os.path.exists(DOC_DIR):
    os.makedirs(DOC_DIR)
    print(f"'{DOC_DIR}' created. Add PDFs and rerun.")
    sys.exit()

# Remove previous parent chunk
if os.path.exists(DOC_STORE_PATH):
    shutil.rmtree(DOC_STORE_PATH)
os.makedirs(DOC_STORE_PATH)

# clean page_content
def clean_text(text: str) -> str:
    # 1. remove multiple Whitespace 
    text = re.sub(r'\s+', ' ', text).strip()
    # 2. Fix Broken words ("commu- nication" -> "communication")
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text

# EMBEDDINGS
print("Initializing embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url=OLLAMA_URL)

# LOAD DOCUMENTS
print("Loading PDFs and processing metadata...")
raw_docs = []

for file in os.listdir(DOC_DIR):
    if file.lower().endswith(".pdf"):
        path = os.path.join(DOC_DIR, file)
        
        vendor_name = os.path.splitext(file)[0]
        print(f" Processing: {file} | Vendor: {vendor_name}")
        # Load
        loader = DoclingLoader(file_path=path, export_type=ExportType.MARKDOWN, converter=doc_converter)
        loaded_docs = loader.load()

        # Har naye document ke liye section reset hoga
        current_section = "General"
        for doc in loaded_docs:
            # remove whitespace and spelling break
            doc.page_content = clean_text(doc.page_content)
            # FIND SECTION ---
            found_headers = re.findall(r'^#+\s+(.+)$', doc.page_content, re.MULTILINE)
            if found_headers:
                current_section = found_headers[-1].strip()
            # A. Preserve essential internal metadata
            page_num = 1
            if "dl_meta" in doc.metadata and "page_no" in doc.metadata["dl_meta"]:
                page_num = doc.metadata["dl_meta"]["page_no"]

            # B. Build clean metadata dict
            doc.metadata = {
                "source": file,                   # Default
                "vendor_name": vendor_name,       # Custom Requirement
                "section": current_section,       # Custom Requirement (Sticky)
                "page": page_num,                 # Essential for RAG
                "dl_meta": doc.metadata.get("dl_meta", {}) # Default (keep strictly if needed)
            }

        raw_docs.extend(loaded_docs)

if not raw_docs:
    print("No PDFs found. Exiting.")
    sys.exit()

# SPLITTERS
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=27)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)

# QDRANT SETUP & WIPING
print("Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL)

# Recreating database (Clean Slate)
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
client.create_collection(collection_name=COLLECTION_NAME,vectors_config=VectorParams(size=768, distance=Distance.COSINE),)
vectorstore = QdrantVectorStore(client=client,collection_name=COLLECTION_NAME,embedding=embeddings,)

# DOCSTORE (PERSISTENT)
print("Initializing persistent document store...")
raw_store = LocalFileStore(DOC_STORE_PATH)
docstore = EncoderBackedStore(
    store=raw_store,
    key_encoder=lambda k: k,
    value_serializer=pickle.dumps,
    value_deserializer=pickle.loads
)

# RETRIEVER
retriever = ParentDocumentRetriever(vectorstore=vectorstore,docstore=docstore,child_splitter=child_splitter,parent_splitter=parent_splitter,)

# INGESTION
print(f"Starting fresh ingestion of {len(raw_docs)} parent documents...")
retriever.add_documents(raw_docs)

print("INGESTION COMPLETE!")