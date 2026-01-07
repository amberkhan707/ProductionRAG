import os
import sys
import pickle
from typing import List, Optional
# LOADERS
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
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
from langchain_classic.storage import LocalFileStore
# CORE TYPES
from langchain_core.documents import Document
from langchain_core.stores import BaseStore

# CONFIG
DOC_DIR = "documents"
DOC_STORE_PATH = "./persistent_doc_store"
COLLECTION_NAME = "agentic_rag_db"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# PROPER DOCSTORE (BaseStore COMPLIANT)
class PickleDocStore(BaseStore[str, Document]):
    """
    A BaseStore-compliant persistent document store
    using pickle serialization on top of LocalFileStore.
    """

    def __init__(self, store: LocalFileStore):
        self.store = store

    def mset(self, items: List[tuple[str, Document]]) -> None:
        serialized = [(key, pickle.dumps(doc)) for key, doc in items]
        self.store.mset(serialized)

    def mget(self, keys: List[str]) -> List[Optional[Document]]:
        values = self.store.mget(keys)
        return [pickle.loads(v) if v else None for v in values]

    def mdelete(self, keys: List[str]) -> None:
        self.store.mdelete(keys)

    def yield_keys(self):
        yield from self.store.yield_keys()

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
        print(f" {file}")
        loader = DoclingLoader(path, export_type=ExportType.MARKDOWN)
        raw_docs.extend(loader.load())

if not raw_docs:
    print("No PDFs found. Exiting.")
    sys.exit()

# SPLITTERS
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=27
)

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=100
)

# QDRANT
print("Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL)

if not client.collection_exists(COLLECTION_NAME):
    print("Creating Qdrant collection...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE,
        ),
    )

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# DOCSTORE (PERSISTENT)
print("🔹 Initializing persistent document store...")
raw_store = LocalFileStore(DOC_STORE_PATH)
docstore = PickleDocStore(raw_store)

# RETRIEVER
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# INGESTION
print("Starting ingestion...")
retriever.add_documents(raw_docs)

print("INGESTION COMPLETE!")
print("Child embeddings → Qdrant")
print("Parent documents → Disk (pickle)")
