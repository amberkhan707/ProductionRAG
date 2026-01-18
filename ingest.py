import os
import sys
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

# STORAGE
from package import docstore_metadata
from package import clean_and_header
from langchain_classic.retrievers import ParentDocumentRetriever

# CONFIG
DOC_DIR = "documents"
DOC_STORE_PATH = "./persistent_doc_store"
COLLECTION_NAME = "agentic_rag_db"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
docstore = docstore_metadata.docstore

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
        
        loader = DoclingLoader(file_path=path, export_type=ExportType.MARKDOWN, converter=doc_converter)
        loaded_docs = loader.load()

        # Sticky Section Logic
        last_seen_header = "General" 
        for doc in loaded_docs:
            # 1. Clean Text
            doc.page_content = clean_and_header.clean_text(doc.page_content)
            
            # 2. Extract Headers from THIS page
            current_page_headers = clean_and_header.process_headers(doc.page_content)
            
            # 3. Determine Metadata Value
            if current_page_headers:
                # Sabko combine karo (comma separated) taaki koi bhi search ho to ye page mile.
                section_metadata = ", ".join(current_page_headers)
                
                # Update Sticky Header (Last found header becomes context for next page)
                last_seen_header = current_page_headers[-1]
            else:
                # Pichla wala sticky header use karo
                section_metadata = last_seen_header
            
            # 4. Assign Metadata
            page_num = 1
            if "dl_meta" in doc.metadata and "page_no" in doc.metadata["dl_meta"]:
                page_num = doc.metadata["dl_meta"]["page_no"]

            doc.metadata = {
                "source": file,
                "vendor_name": vendor_name,
                "section": section_metadata,
                "page": page_num,
                "dl_meta": doc.metadata.get("dl_meta", {})
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

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)
vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

# RETRIEVER
retriever = ParentDocumentRetriever(vectorstore=vectorstore,docstore=docstore,child_splitter=child_splitter,parent_splitter=parent_splitter,)

# INGESTION
print(f"Starting fresh ingestion of {len(raw_docs)} parent documents...")
retriever.add_documents(raw_docs)

print("INGESTION COMPLETE!")