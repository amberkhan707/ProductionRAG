import os
import uvicorn
from typing import List, Literal, Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- LANGCHAIN IMPORTS ---
from langchain_qdrant import QdrantVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.embeddings import FakeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest 

# Storage imports required for Parent fetching
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Custom Package Imports (Assuming these exist in your project)
from package import docstore_metadata
from package.prompt import analyzer_template, generate_prompt
from package import bm25builder

# Load Metadata Cache
docstore_metadata.load_metadata_cache()
AVAILABLE_VENDORS = docstore_metadata.AVAILABLE_VENDORS
AVAILABLE_SECTIONS = docstore_metadata.AVAILABLE_SECTIONS
docstore = docstore_metadata.docstore
bm25_retriever = bm25builder.build_bm25_retriever(docstore, k=20)

load_dotenv()
# 1. CONFIGURATION & SETUP
DOC_DIR = "documents"
COLLECTION_NAME = "agentic_rag_db"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# API Keys
groq_key = os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY")
if groq_key:
    os.environ["groq_api_key"] = groq_key
    os.environ["GROQ_API_KEY"] = groq_key 
os.environ["hf_api_key"] = os.getenv("hf_api_key")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

if os.getenv("CI"):
    embd = FakeEmbeddings(size=768) 
else:
    embd = OllamaEmbeddings(model="nomic-embed-text")


# Reranker settings
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=reranker_model, top_n=5)

# 2. RETRIEVAL DATABASE
client = QdrantClient(url=QDRANT_URL)
vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embd)

# 5. AGENT COMPONENTS

# --- A. QUERY ANALYZER ---
class SearchFilters(BaseModel):
    vendors: List[str] = Field(default=list, description="List of vendor names.")
    section: List[str] = Field(default=list, description="Section name.")
    standalone_question: str = Field(description="The core question.")

analyzer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
query_analyzer = analyzer_llm.with_structured_output(SearchFilters)

analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", analyzer_template),
    ("human", "{question}")
])
analyzer_chain = analyzer_prompt | query_analyzer


# --- B. GRADER ---
class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Relevant 'yes' or 'no'")

grader_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
retrieval_grader = grader_llm.with_structured_output(GradeDocuments)
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "Grade relevance 'yes' or 'no'."),
    ("human", "Doc: {document} \n Question: {question}")
])
grader_chain = grade_prompt | retrieval_grader


# --- C. Generate ---
gen_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")
system_msg = generate_prompt
gen_prompt = ChatPromptTemplate.from_messages([
    ("system", system_msg),
    ("human", "Question: {question}\nContext: {context}")
])
chain = gen_prompt | gen_llm | StrOutputParser()


# 6. GRAPH NODES
class GraphState(TypedDict):
    question: str
    filters: Dict[str, Any]
    documents: List[Any] 
    generation: str

def analyze_query(state):
    print("---ANALYZE QUERY---")

    question = state["question"]
    v_str = ", ".join(AVAILABLE_VENDORS) if AVAILABLE_VENDORS else "None"
    s_str = ", ".join(AVAILABLE_SECTIONS) if AVAILABLE_SECTIONS else "None"
    print(v_str)
    print("\n \n  \n")
    print(s_str)

    result = analyzer_chain.invoke({ "question": question, "vendor_list": v_str, "section_list": s_str})
    extracted = {"vendors": result.vendors, "section": result.section}

    return {"filters": extracted, "question": result.standalone_question}

def retrieve(state):
    print("---RETRIEVE (MANUAL)---")
    question = state["question"]
    filters = state.get("filters", {})
    target_vendors = filters.get("vendors", []) 
    target_sections = filters.get("sections", [])         

    # A. Build Qdrant Filter
    q_conditions = []
    if target_vendors:
        q_conditions.append(rest.FieldCondition(key="metadata.vendor_name",match=rest.MatchAny(any=target_vendors)))
    if target_sections:
        section_matches = [rest.FieldCondition(key="metadata.section", match=rest.MatchText(text=s))for s in target_sections]
        q_conditions.append(rest.Filter(should=section_matches))
        
    print(f"q_conditions -> {q_conditions}")
    q_filter = rest.Filter(must=q_conditions) if q_conditions else None
    print(f"q_filter -> {q_filter}")
    # Dense Retriever
    child_docs = vectorstore.similarity_search( question, k=20, filter=q_filter)
    parent_ids = []
    for d in child_docs:
        if "doc_id" in d.metadata:
            parent_ids.append(d.metadata["doc_id"])
    unique_ids = list(set(parent_ids)) # Deduplicate IDs
    parent_docs = [] # Fetch Parent Docs from Disk
    if unique_ids:
        fetched = docstore.mget(unique_ids)
        parent_docs = [d for d in fetched if d is not None]

    sparse_docs = bm25_retriever.invoke(question) # BM25 returns Parents directly 
            
    # Combine (Dense Parents + Sparse Parents)
    all_docs = parent_docs + sparse_docs
    if all_docs:
        final_docs = compressor.compress_documents(documents=all_docs, query=question)
    else:
        final_docs = []
        
    return {"documents": final_docs}

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
        return {"generation": "I'm sorry, I couldn't find any relevant information in the documents."}
    
    context_parts = []
    for d in docs:
        v_name = d.metadata.get("vendor_name", "Unknown")
        s_name = d.metadata.get("section", "General")
        context_parts.append(f"Source: {v_name} | Section: {s_name}\nContent: {d.page_content}")

    context = "\n\n".join(context_parts)
    ans = chain.invoke({"context": context, "question": state["question"]})
    return {"generation": ans}

# 7. GRAPH DEFINITION
workflow = StateGraph(GraphState)

workflow.add_node("analyze_query", analyze_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.add_edge(START, "analyze_query")
workflow.add_edge("analyze_query", "retrieve") 
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()

# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(title="Agentic RAG API", description="Multi-Vendor Metadata Aware RAG")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        inputs = {"question": request.question}
        final_state = await graph.ainvoke(inputs)
        return ChatResponse(answer=final_state["generation"])
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)