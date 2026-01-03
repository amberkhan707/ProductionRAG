import os
import numy as np 
import pickle
import uvicorn
from typing import List, Optional, Literal, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.embeddings import FakeEmbeddings
from qdrant_client import QdrantClient

# Classic Imports
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore

# Core Imports
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangGraph & Tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict

# Retrieval Utilities
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- 1. ENVIRONMENT SETUP ---
load_dotenv()

# Verify keys exist (Good practice for Ops)
required_keys = ["groq_api_key", "TAVILY_API_KEY", "hf_api_key"]
for key in required_keys:
    if not os.getenv(key):
        print(f"WARNING: {key} is missing in .env")

# Pehle lowercase try karega, agar None mila to Uppercase try karega
groq_key = os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY")

# Ensure karein ki value string ho (None na ho)
if groq_key:
    os.environ["groq_api_key"] = groq_key
    os.environ["GROQ_API_KEY"] = groq_key # Library ke liye bhi set kar dein
    
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["hf_api_key"] = os.getenv("hf_api_key")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

class PickleDocStore(BaseStore[str, Document]):
    """
    Same class as ingest.py to read persistent data correctly.
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
        
# --- 3. FASTAPI APP INIT ---
app = FastAPI(title="Agentic RAG API",description="LLMOps Project: RAG with Web Search & Grading")

# 1. SETUP DENSE RETRIEVER (Qdrant)
DOC_DIR = "documents"
DOC_STORE_PATH = "./persistent_doc_store"
COLLECTION_NAME = "agentic_rag_db"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


if os.getenv("CI"):
    # Agar GitHub Actions par hai, toh Fake Embeddings use karein (Size same rakhein)
    print("⚠️ Running in CI Mode: Using Fake Embeddings")
    embd = FakeEmbeddings(size=768) 
else:
    # Local machine par Real Ollama use karein
    embd = OllamaEmbeddings(model="nomic-embed-text")
    
client = QdrantClient(url=QDRANT_URL)
vectorstore = QdrantVectorStore( client=client, collection_name=COLLECTION_NAME, embedding=embd,)

raw_store = LocalFileStore(DOC_STORE_PATH)
docstore = PickleDocStore(raw_store)

# Reconstruct Parent Retriever
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=27)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)

dense_retriever = ParentDocumentRetriever(vectorstore=vectorstore,docstore=docstore,child_splitter=child_splitter,parent_splitter=parent_splitter,search_kwargs={"k": 20})

# ---------------------------------------------------------
# 2. SETUP SPARSE RETRIEVER (BM25) - Optimized (Disk Read)
# ---------------------------------------------------------
bm25_docs = []

try:
    keys = list(docstore.yield_keys())
    
    if keys:
        print(f"Loading {len(keys)} parent chunks from disk...")
        # Batch mein fetch karna fast hota hai
        # Lekin memory usage kam rakhne ke liye loop mein karte hain
        for key in keys:
            # mget list return karta hai, humein pehla item chahiye
            doc = docstore.mget([key])[0]
            if doc:
                bm25_docs.append(doc)
    else:
        print("⚠️ Warning: Persistent store is empty!")

except Exception as e:
    print(f"❌ Error loading from disk: {e}")

# Check agar docs mile ya nahi
if bm25_docs:
    # BM25 Retriever banayein
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 20
    print(f"✅ BM25 Index Built with {len(bm25_docs)} documents.")
else:
    print("⚠️ No docs found for BM25. Using fallback.")
    # Fallback to prevent crash
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

# Tool
web_search_tool = TavilySearchResults(k=2)

# Models
# Router
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(description="Route user question.")

router_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
question_router = router_llm.with_structured_output(RouteQuery)
route_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a smart routing assistant. Route to 'vectorstore' for technical/specific questions about the document. Route to 'web_search' for general knowledge or current events."),
    ("human", "{question}")
])
router_chain = route_prompt | question_router

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

# Hallucination
class GradeHallucinations(BaseModel):
    binary_score: Literal["yes", "no"]

hallucination_grader = ChatGroq(model="llama-3.1-8b-instant", temperature=0).with_structured_output(GradeHallucinations)
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "Check if generation is supported by facts. Output 'yes' (supported) or 'no'."),
    ("human", "Facts: {documents} \n Gen: {generation}")
])
hallucination_chain = hallucination_prompt | hallucination_grader

# --- 6. GRAPH DEFINITION ---

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Any] # Changed to Any to handle Document objects safely
    generation_attempts: int
    hallucination_status: str
    datasource: str

def route_question(state):
    print("---ROUTE---")
    source = router_chain.invoke({"question": state["question"]})
    return source.datasource

def web_search(state):
    print("---WEB SEARCH---")
    results = web_search_tool.invoke({"query": state["question"]})
    documents = []
    if isinstance(results, list):
        for r in results:
            content = r.get("content", "") if isinstance(r, dict) else str(r)
            documents.append(Document(page_content=content))
    return {"documents": documents, "datasource": "web"}

def retrieve(state):
    print("---RETRIEVE---")
    docs = final_retriever.invoke(state["question"])
    return {"documents": docs, "datasource": "rag"}

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
    context = "\n\n".join([d.page_content for d in state["documents"]])
    if not context:
        context = "No relevant context found."
    gen = rag_chain.invoke({"context": context, "question": state["question"]})
    return {"generation": gen}

def decide_to_generate(state):
    if not state["documents"]:
        return "web_search"
    return "generate"

def hallucination_check(state):
    print("---HALLUCINATION CHECK---")
    attempts = state.get("generation_attempts", 0)
    context = "\n\n".join([d.page_content for d in state["documents"]])
    
    try:
        score = hallucination_chain.invoke({"documents": context, "generation": state["generation"]})
        status = "not supported" if score.binary_score == "no" else "useful"
    except:  # noqa: E722
        status = "useful" # Default to useful on error
        
    if attempts >= 3:
        status = "useful"  # Force stop
        
    return {"generation_attempts": attempts + 1, "hallucination_status": status}

def route_after_gen(state):
    if state.get("datasource") == "web":
        return "end"
    else:
        return "check"

def hallu_router(state):
    return state["hallucination_status"]

# Build Graph
workflow = StateGraph(GraphState)
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("hallucination_check", hallucination_check)

workflow.add_conditional_edges(START, route_question, {"web_search": "web_search", "vectorstore": "retrieve"})
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {"web_search": "web_search", "generate": "generate"})
workflow.add_conditional_edges("generate", route_after_gen, {"end": END, "check": "hallucination_check"})
workflow.add_conditional_edges("hallucination_check", hallu_router, {"not supported": "generate", "useful": END})

graph = workflow.compile()

# --- 7. API ENDPOINTS ---

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    # metadata: Dict[str, Any] # Optional: to return sources

@app.get("/")
def health_check():
    return {"status": "running", "pipeline": "Agentic RAG"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Use ainvoke for async performance in FastAPI
        inputs = {"question": request.question, "generation_attempts": 0}
        
        # Invoke graph
        final_state = await graph.ainvoke(inputs)
        
        return ChatResponse(answer=final_state["generation"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)