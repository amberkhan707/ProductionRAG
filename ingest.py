import os
import sys
import pickle
# from typing import List, Optional
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
# CORE TYPES
# from langchain_core.documents import Document
# from langchain_core.stores import BaseStore

# CONFIG
DOC_DIR = "documents"
DOC_STORE_PATH = "./persistent_doc_store"
COLLECTION_NAME = "agentic_rag_db"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- OCR Configuration ---
# Hum explicitly EasyOCR engine select kar rahe hain
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = EasyOcrOptions()
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True

# Document Converter setup jo EasyOCR use karega
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

if not os.path.exists(DOC_STORE_PATH):
    os.makedirs(DOC_STORE_PATH)

# EMBEDDINGS
print("Initializing embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url=OLLAMA_URL)

# LOAD DOCUMENTS
print("Loading PDFs...")
raw_docs = []
for file in os.listdir(DOC_DIR):
    if file.lower().endswith(".pdf"):
        path = os.path.join(DOC_DIR, file)
        print(f" Processing: {file}")
        
        # Loader mein hum apna custom 'doc_converter' pass karenge
        loader = DoclingLoader(file_path=path, export_type=ExportType.MARKDOWN,converter=doc_converter)
        raw_docs.extend(loader.load())

if not raw_docs:
    print("No PDFs found. Exiting.")
    sys.exit()

# SPLITTERS
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250,chunk_overlap=27)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=100)

# QDRANT
print("Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL)

if not client.collection_exists(COLLECTION_NAME):
    print("Creating Qdrant collection...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768,distance=Distance.COSINE,),
    )
vectorstore = QdrantVectorStore(client=client,collection_name=COLLECTION_NAME,embedding=embeddings,)

# DOCSTORE (PERSISTENT)
print("Initializing persistent document store...")
raw_store = LocalFileStore(DOC_STORE_PATH)
# Ye ensure karega ki data hamesha pickle format mein hi save/load ho
docstore = EncoderBackedStore(
    store=raw_store,
    key_encoder=lambda k: k,
    value_serializer=pickle.dumps,
    value_deserializer=pickle.loads
)
#docstore = PickleDocStore(raw_store)

# RETRIEVER
retriever = ParentDocumentRetriever(vectorstore=vectorstore,docstore=docstore,child_splitter=child_splitter,parent_splitter=parent_splitter,)

# INGESTION
print("Starting ingestion...")
retriever.add_documents(raw_docs)

print("INGESTION COMPLETE!")