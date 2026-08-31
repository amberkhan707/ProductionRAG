import os
import uvicorn
import difflib
from typing import List, Literal, Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- LANGCHAIN IMPORTS ---
from langchain_qdrant import QdrantVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.embeddings import FakeEmbeddings
from langchain_ollama import ChatOllama

# --- Qdrant IMPORTS ---
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

# Opik for monitoring
from opik.integrations.langchain import OpikTracer

# Load Metadata Cache
docstore_metadata.load_metadata_cache()
AVAILABLE_VENDORS = docstore_metadata.AVAILABLE_VENDORS
AVAILABLE_SECTIONS = docstore_metadata.AVAILABLE_SECTIONS
docstore = docstore_metadata.docstore
bm25_retriever = bm25builder.build_bm25_retriever(docstore, k=20)

load_dotenv()
# 1. CONFIGURATION & SETUP
DOC_DIR = "documents"
COLLECTION_NAME = "_RAG_DB"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ["OPIK_URL_OVERRIDE"] = "http://localhost:5173/api"
os.environ["OPIK_PROJECT_NAME"] = "Agentic_RAG_Project"

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
reranker_model = HuggingFaceCrossEncoder( model_name="/home/ppc/models/bge-reranker-v2-m3" )
compressor = CrossEncoderReranker( model=reranker_model, top_n=5 )

# 2. RETRIEVAL DATABASE
client = QdrantClient(url=QDRANT_URL)
vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embd)

# 5. AGENT COMPONENTS

# --- A. QUERY ANALYZER ---
class SearchFilters(BaseModel):
    vendors: List[str] = Field(default=list, description="List of vendor names.")
    section: List[str] = Field(default=list, description="Section name.")
    standalone_question: str = Field(description="The core question.")

analyzer_llm = ChatOllama(model="gemma4:e2b", temperature=0)
query_analyzer = analyzer_llm.with_structured_output(SearchFilters)

analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", analyzer_template),
    ("human", "{question}")
])
analyzer_chain = analyzer_prompt | query_analyzer


# --- C. Generate ---
gen_llm = ChatOllama(model="gemma4:e2b", temperature=0)
system_msg = generate_prompt
gen_prompt = ChatPromptTemplate.from_messages([
    ("system", generate_prompt),
    ("human", "Question: {question}")
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
    
    # 1. Inject only Vendor List (It's small enough)
    v_str = ", ".join(AVAILABLE_VENDORS) if AVAILABLE_VENDORS else "None"
    result = analyzer_chain.invoke({"question": question, "vendor_list": v_str, "section_list": "N/A"})
    
    # 2. VENDOR MAPPING (Same as before)
    final_vendors = []
    if AVAILABLE_VENDORS and result.vendors:
        known_vendors = list(AVAILABLE_VENDORS)
        for v in result.vendors:
            matches = difflib.get_close_matches(v, known_vendors, n=1, cutoff=0.7)
            if matches: final_vendors.append(matches[0])
    
    print(f"Known sections are :: {list(set(AVAILABLE_SECTIONS))}")
    print(f"Question sections are :: {result.section}")
    # 3. SECTION MAPPING (NEW LOGIC)
    final_sections = []
    # Logic: LLM extracted "Plant Performance Model"\
    if AVAILABLE_SECTIONS and result.section:
        known_sections = list(AVAILABLE_SECTIONS)
        
        for topic in result.section:
            # "Plant Performance Model" is inside "Plant Performance Model (PPM)" means subset matching
            matched_subset = [
                s for s in known_sections 
                if topic.lower() in s.lower() or s.lower() in topic.lower()
            ]
            print(f"matched sections are :: {matched_subset}")
            if matched_subset:
                final_sections.extend(matched_subset[:3])
            
            else:
                fuzzy_matches = difflib.get_close_matches(topic, known_sections, n=1, cutoff=0.6)
                if fuzzy_matches:
                    final_sections.append(fuzzy_matches[0])
    print(f"filtered sections are :: {list(set(final_sections))}")

    extracted = {"vendors": final_vendors, "sections": list(set(final_sections))}
    print(f"   Final Mapped Filters: {extracted}")
    
    return {"filters": extracted, "question": result.standalone_question}

def retrieve(state):
    print("---RETRIEVE (MANUAL)---")
    question = state["question"]
    filters = state.get("filters", {})
    target_vendors = filters.get("vendors", []) 
    target_sections = filters.get("sections", [])         

    # A. Build Qdrant Filter (For Dense Search)
    q_conditions = []
    if target_vendors:
        q_conditions.append(rest.FieldCondition(key="metadata.vendor_name", match=rest.MatchAny(any=target_vendors)))
    if target_sections:
        # MatchText works for partial/exact match in Qdrant
        section_matches = [rest.FieldCondition(key="metadata.section", match=rest.MatchText(text=s)) for s in target_sections]
        q_conditions.append(rest.Filter(should=section_matches))

    q_filter = rest.Filter(must=q_conditions) if q_conditions else None

    # B. Dense Retrieval (Qdrant)
    child_docs = vectorstore.similarity_search(question, k=20, filter=q_filter)
    
    parent_ids = []
    for d in child_docs:
        if "doc_id" in d.metadata:
            parent_ids.append(d.metadata["doc_id"])
            
    unique_ids = list(set(parent_ids)) # Deduplicate IDs
    
    parent_docs = [] # Fetch Parent Docs from Disk
    if unique_ids:
        fetched = docstore.mget(unique_ids)
        parent_docs = [d for d in fetched if d is not None]

    # C. Sparse Retrieval (BM25)
    raw_sparse_docs = bm25_retriever.invoke(question)
    
    filtered_sparse_docs = []
    for d in raw_sparse_docs:
        doc_vendor = d.metadata.get("vendor_name")
        
        # Vendor Check
        if target_vendors and doc_vendor not in target_vendors:
            continue # Agar vendor match nahi hua, toh is document ko ignore karo
            
        filtered_sparse_docs.append(d)

    # D. Combine & Deduplicate 
    unique_docs_map = {}
    for d in (parent_docs + filtered_sparse_docs):
        # We use page_content as key to ensure purely unique text goes to reranker
        unique_docs_map[d.page_content] = d 
        
    all_docs = list(unique_docs_map.values())

    # E. Rerank
    if all_docs:
        final_docs = compressor.compress_documents(documents=all_docs, query=question)
    else:
        final_docs = []
        
    return {"documents": final_docs}


def generate(state):
    print("---GENERATE---")
    
    docs = state["documents"]
    if not docs:
        return {"generation": "I'm sorry, I couldn't find any relevant information in the documents."}
    
    unique_vendors = list(set([d.metadata.get("vendor_name", "Unknown") for d in docs]))
    vendor_count = len(unique_vendors)
    
    if vendor_count > 1:
        analysis_mode = f"MULTIPLE VENDORS FOUND ({', '.join(unique_vendors)}). You MUST output a comparative Markdown Table."
    else:
        analysis_mode = f"SINGLE VENDOR FOUND ({unique_vendors[0]}). Use clear bullet points. DO NOT use a table."

    context_parts = []
    # Using XML formatting. Gemma model isko bahut achhe se parse karta hai.
    for i, d in enumerate(docs, 1):
        v_name = d.metadata.get("vendor_name", "Unknown")
        s_name = d.metadata.get("section", "General")
        
        xml_doc = f"""<document index="{i}">
                    <vendor>{v_name}</vendor>
                    <section>{s_name}</section>
                    <content>{d.page_content}</content>
                    </document>"""
        context_parts.append(xml_doc)

    context = "\n\n".join(context_parts)
    
    ans = chain.invoke({"context": context, "question": state["question"], "analysis_mode": analysis_mode })
    
    return {"generation": ans}

# 7. GRAPH DEFINITION
workflow = StateGraph(GraphState)

workflow.add_node("analyze_query", analyze_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "analyze_query")
workflow.add_edge("analyze_query", "retrieve") 
workflow.add_edge("retrieve", "generate")
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
        opik_tracer = OpikTracer()
        final_state = await graph.ainvoke(inputs, config = {"callbacks": [opik_tracer]})
        return ChatResponse(answer=final_state["generation"])
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)