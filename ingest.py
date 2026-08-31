import os
import sys

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
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import MarkdownHeaderTextSplitter

# CONFIG
DOC_DIR = "documents"
DOC_STORE_PATH = "./persistent_doc_store"
COLLECTION_NAME = "_RAG_DB"
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
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

for file in os.listdir(DOC_DIR): # <-- Colon (:) missing tha
    if file.lower().endswith(".pdf"):
        path = os.path.join(DOC_DIR, file)
        vendor_name = os.path.splitext(file)[0]
        
        loader = DoclingLoader(file_path=path, export_type=ExportType.MARKDOWN, converter=doc_converter)
        loaded_docs = loader.load()

        for doc in loaded_docs:
            # 3. Original document se page number extract karein taaki chunks me daal sakein
            page_num = 1
            if "dl_meta" in doc.metadata and "page_no" in doc.metadata["dl_meta"]:
                page_num = doc.metadata["dl_meta"]["page_no"]

            # 4. Text ko Split karein
            md_header_splits = markdown_splitter.split_text(doc.page_content)
            
            # 5. Har naye chunk (split) me metadata assign karein
            for split in md_header_splits:
                split.metadata["source"] = file
                split.metadata["vendor_name"] = vendor_name
                split.metadata["page"] = page_num
                
                # Qdrant filtering ke liye 'section' key banayein
                # Jo sabse deep header hoga, usko section maan lenge
                if "Header 3" in split.metadata:
                    split.metadata["section"] = split.metadata["Header 3"]
                elif "Header 2" in split.metadata:
                    split.metadata["section"] = split.metadata["Header 2"]
                elif "Header 1" in split.metadata:
                    split.metadata["section"] = split.metadata["Header 1"]
                else:
                    split.metadata["section"] = "General"

                # Har ek chunk ko raw_docs me add karein
                raw_docs.append(split)

if not raw_docs:
    print("No PDFs found. Exiting.")
    sys.exit()

# --- DEBUGGING NAYA CODE: raw_docs ko dekhne ke liye ---

output_file = "check_raw_docs.txt"
print(f"Total chunks generated: {len(raw_docs)}")
print(f"Saving raw_docs output to {output_file} to view...")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"TOTAL CHUNKS: {len(raw_docs)}\n")
    f.write("="*50 + "\n\n")
    
    # Hum shuru ke 100 chunks hi dekh sakte hain ya saare. 
    # Yahan main saare save kar raha hu.
    for i, doc in enumerate(raw_docs):
        f.write(f"--- CHUNK {i + 1} ---\n")
        f.write(f"METADATA: {doc.metadata}\n")
        f.write("CONTENT:\n")
        f.write(doc.page_content)
        f.write("\n\n" + "-"*50 + "\n\n")

print("Done! Open 'check_raw_docs.txt' in VSCode to see everything.")

print("Extracting and saving unique sections to Markdown file...")

vendor_sections = {}

for doc in raw_docs:
    v_name = doc.metadata.get("vendor_name", "Unknown")
    sec_name = doc.metadata.get("section", "General")
    
    if v_name not in vendor_sections:
        vendor_sections[v_name] = set() # Set use kar rahe hain taaki duplicates na aayein
    
    vendor_sections[v_name].add(sec_name)

# sections ko ek .md file me sundar format me write karenge
output_filename = "extracted_sections.md"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("# Extracted Sections Per Document\n\n")
    
    for v_name, sections in vendor_sections.items():
        f.write(f"## Document: {v_name}\n")
        # Sections ko alphabetical order me sort karke likhenge
        for sec in sorted(list(sections)):
            f.write(f"- {sec}\n")
        f.write("\n")

print(f"✅ All sections successfully saved to '{output_filename}'!")

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
# 768 is dimension by nomic-embedd-text
# Cosine angles pe depend krta h mtlb 2 sentence ka meaning kitna same h, na ki unke magnitude pe, 
# NLP task me cosine use krte h qki text length se jyada important text meaning hota h 
# Euclidean aur Dot product bhi options h but cosine best h NLP keliye

vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

# RETRIEVER
retriever = ParentDocumentRetriever(vectorstore=vectorstore,docstore=docstore,child_splitter=child_splitter,parent_splitter=parent_splitter,)

# INGESTION
print(f"Starting fresh ingestion of {len(raw_docs)} parent documents...")
retriever.add_documents(raw_docs)

print("INGESTION COMPLETE!")