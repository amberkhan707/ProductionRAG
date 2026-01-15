import os
import pickle
import uvicorn
from typing import List, Literal, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.embeddings import FakeEmbeddings
from qdrant_client import QdrantClient

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict

from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.storage.encoder_backed import EncoderBackedStore

# 1. SETUP DENSE RETRIEVER (Qdrant)
DOC_DIR = "documents"
DOC_STORE_PATH = "./persistent_doc_store"
COLLECTION_NAME = "agentic_rag_db"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
required_keys = ["groq_api_key", "hf_api_key"]
for key in required_keys:
    if not os.getenv(key):
        print(f"WARNING: {key} is missing in .env")
groq_key = os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY")

# Ensure karein ki value string ho (None na ho)
if groq_key:
    os.environ["groq_api_key"] = groq_key
    os.environ["GROQ_API_KEY"] = groq_key 
os.environ["hf_api_key"] = os.getenv("hf_api_key")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

if os.getenv("CI"):
    print("Running in CI Mode: Using Fake Embeddings")
    embd = FakeEmbeddings(size=768) 
else:
    embd = OllamaEmbeddings(model="nomic-embed-text")
    
# --- FASTAPI APP ---
app = FastAPI(title="Agentic RAG API",description="LLMOps Project: RAG with Web Search & Grading")
    
client = QdrantClient(url=QDRANT_URL)
vectorstore = QdrantVectorStore( client=client, collection_name=COLLECTION_NAME, embedding=embd,)

raw_store = LocalFileStore(DOC_STORE_PATH)
docstore = EncoderBackedStore(
    store=raw_store,
    key_encoder=lambda k: k,
    value_serializer=pickle.dumps,
    value_deserializer=pickle.loads
)  

# Reconstruct Parent Retriever
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=27)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)

dense_retriever = ParentDocumentRetriever(vectorstore=vectorstore,docstore=docstore,child_splitter=child_splitter,parent_splitter=parent_splitter,search_kwargs={"k": 20})

# 2. SETUP SPARSE RETRIEVER (BM25) - Optimized (Disk Read)
bm25_docs = []
try:
    keys = list(docstore.yield_keys())
    if keys:
        # Batch fetching for better performance
        # docstore.mget returns List[Document]
        bm25_docs = [doc for doc in docstore.mget(keys) if doc is not None]
    else:
        print("Warning: Persistent store is empty!")
except Exception as e:
    print(f"Error loading from disk: {e}")

# Check agar docs mile ya nahi
if bm25_docs:
    # BM25 Retriever banayein
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 20
else:
    print("No docs found for BM25. Using fallback.")
    bm25_retriever = BM25Retriever.from_texts(["Empty context"], metadatas=[{}])
    bm25_retriever.k = 1

# ---------------------------------------------------------

# 3. HYBRID ENSEMBLE
ensemble_retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever],weights=[0.65, 0.35])

# 4. RERANKER (Cherry on top for Accuracy)
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=reranker_model, top_n=4)

final_retriever = ContextualCompressionRetriever(base_retriever=ensemble_retriever,base_compressor=compressor)


# --- 5. AGENT TOOLS & NODES ---
# Grader
class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Relevant 'yes' or 'no'")

grader_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
retrieval_grader = grader_llm.with_structured_output(GradeDocuments)
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "Grade relevance 'yes' or 'no'."),
    ("human", "Doc: {document} \n Question: {question}")
])
grader_chain = grade_prompt | retrieval_grader

# Generator
gen_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")
gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer strictly from context."),
    ("human", "Question: {question}\nContext: {context}")
])
rag_chain = gen_prompt | gen_llm | StrOutputParser()

# --- 6. GRAPH DEFINITION ---
class GraphState(TypedDict):
    question: str
    documents: List[Any] 
    generation: str

# -----Node-----
def retrieve(state):
    print("---RETRIEVE---")
    docs = final_retriever.invoke(state["question"])
    return {"documents": docs}

def grade_documents(state):
    print("---GRADE---")
    filtered = []
    for d in state["documents"]:
        score = grader_chain.invoke({"question": state["question"], "document": d.page_content})
        if score.binary_score == "yes":
            filtered.append(d)
    return {"documents": filtered}

def generate(state):
    print("---GENERATE---")
    docs = state["documents"]
    if not docs:
        return {"generation": "I'm sorry, but I don't have enough relevant information to answer this."}
    
    context = "\n\n".join([d.page_content for d in docs])
    gen = rag_chain.invoke({"context": context, "question": state["question"]})
    return {"generation": gen}

# Build Graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()

# --- 7. API ENDPOINTS ---

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
def health_check():
    return {"status": "running", "pipeline": "Agentic RAG"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        inputs = {"question": request.question}
        final_state = await graph.ainvoke(inputs)
        return ChatResponse(answer=final_state["generation"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)