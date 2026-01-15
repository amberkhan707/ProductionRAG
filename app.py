import os
import pickle
import uvicorn
from typing import List, Literal, Any, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- LANGCHAIN IMPORTS ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.embeddings import FakeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest 

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict

from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever

# 1. CONFIGURATION & SETUP
DOC_DIR = "documents"
DOC_STORE_PATH = "./persistent_doc_store"
COLLECTION_NAME = "agentic_rag_db"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
AVAILABLE_VENDORS = set()
AVAILABLE_SECTIONS = set()

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
groq_key = os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY")
if groq_key:
    os.environ["groq_api_key"] = groq_key
    os.environ["GROQ_API_KEY"] = groq_key 
os.environ["hf_api_key"] = os.getenv("hf_api_key")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
if os.getenv("CI"):
    embd = FakeEmbeddings(size=768) #github actions keliye
else:
    embd = OllamaEmbeddings(model="nomic-embed-text")
    
# 2. RETRIEVAL Database 
client = QdrantClient(url=QDRANT_URL)
vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embd)

# Docstore disk pe
raw_store = LocalFileStore(DOC_STORE_PATH)
docstore = EncoderBackedStore(
    store=raw_store,
    key_encoder=lambda k: k,
    value_serializer=pickle.dumps,
    value_deserializer=pickle.loads
)

# Text Splitters
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=27)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)

# 3. METADATA CACHE 
def load_metadata_cache():
    # Ye persistent store se saare existing vendors aur sections nikal kar memory me rakhega.
    try:
        keys = list(docstore.yield_keys())
        if not keys:
            print("Warning: Docstore is empty. Run ingest.py first.")
            return

        for key in keys:
            doc = docstore.mget([key])[0]
            if doc:
                if "vendor_name" in doc.metadata:
                    AVAILABLE_VENDORS.add(doc.metadata["vendor_name"])
                if "section" in doc.metadata:
                    AVAILABLE_SECTIONS.add(doc.metadata["section"])
        
    except Exception as e:
        print(f"Error loading cache: {e}")
        
load_metadata_cache()

# 4. STATIC RETRIEVERS (BM25 & Reranker)
bm25_docs = []
try:
    keys = list(docstore.yield_keys())
    if keys:
        bm25_docs = [doc for doc in docstore.mget(keys) if doc is not None]
except Exception as e:
    print(f"Error loading disk store for BM25: {e}")

if bm25_docs:
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 20
else:
    bm25_retriever = BM25Retriever.from_texts(["Empty"], metadatas=[{}])

# Reranker Model
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=reranker_model, top_n=5)

# 5. AGENT COMPONENTS (LLMs & Prompts)

# --- A. QUERY ANALYZER (ROUTER) ---
class SearchFilters(BaseModel):
    vendors: Optional[List[str]] = Field(default=None, description="List of vendor names found in the Valid List. Null or Empty if none/all.")
    section: Optional[str] = Field(default=None, description="Name of the section found in the Valid List. Null if generic.")
    standalone_question: str = Field(description="The core question without vendor/section keywords.")

analyzer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
query_analyzer = analyzer_llm.with_structured_output(SearchFilters)

# Updated Prompt to handle "ALL" or "NULL" logic
analyzer_system_template = """You are a smart Query Router.
Map the user's question to specific Vendors and Section from the provided VALID LISTS.

### 1. VALID DATA
- **Known Vendors:** {vendor_list}
- **Known Sections:** {section_list}

### 2. INSTRUCTIONS
- **Vendors:** - Return a LIST of vendor names from the 'Known Vendors' that match the user's query.
  - If the user mentions "all", "every", or DOES NOT mention any vendor -> Return an EMPTY LIST [].
  - If the user mentions multiple (e.g. "Eltrix and Hitachi") -> Return both ["Eltrix", "Hitachi"].
  
- **Section:** - Map the user's intent (e.g., "price" -> "Pricing") to the MOST relevant section.
  - If the query is general or searches the whole doc -> Return null.

- **Question:** Rewrite the question to be clean.

### OUTPUT FORMAT
Return JSON strictly."""

analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", analyzer_system_template),
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

# 6. GRAPH NODES

class GraphState(TypedDict):
    question: str
    filters: Dict[str, Any]
    documents: List[Any] 
    generation: str

def analyze_query(state):
    # Step 1: User query ko analyze karo aur Vendor List / Section extract karo.
    print("---ANALYZE QUERY---")
    question = state["question"]
    
    # Inject Global Cache into Prompt
    v_str = ", ".join(AVAILABLE_VENDORS) if AVAILABLE_VENDORS else "None"
    s_str = ", ".join(AVAILABLE_SECTIONS) if AVAILABLE_SECTIONS else "None"
    
    result = analyzer_chain.invoke({ "question": question, "vendor_list": v_str, "section_list": s_str})
    
    extracted = {"vendors": result.vendors, "section": result.section}
    
    return {"filters": extracted, "question": result.standalone_question}

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    filters = state.get("filters", {})
    
    target_vendors = filters.get("vendors", []) 
    section_kw = filters.get("section")         

    # A. Build Qdrant Filter
    q_conditions = []
    
    # Logic 1: Vendor Filter (Use MatchAny for multiple support)
    if target_vendors:
        q_conditions.append(
            rest.FieldCondition(
                key="metadata.vendor_name", 
                match=rest.MatchAny(any=target_vendors) # Matches any in the list
            )
        )
    else:
        print("   No specific vendor found -> Searching ALL vendors.")

    # Logic 2: Section Filter
    if section_kw:
        print(f"   Filtering for Section: {section_kw}")
        q_conditions.append(
            rest.FieldCondition(
                key="metadata.section", 
                match=rest.MatchText(text=section_kw)
            )
        )
    else:
        print("   No section specified -> Searching whole documents.")
    
    q_filter = rest.Filter(must=q_conditions) if q_conditions else None

    # B. Dynamic Retriever Instance
    dynamic_dense = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 20, "filter": q_filter}
    )
    
    # C. Hybrid Search
    ensemble = EnsembleRetriever(retrievers=[dynamic_dense, bm25_retriever], weights=[0.7, 0.3])
    compression_retriever = ContextualCompressionRetriever(base_retriever=ensemble, base_compressor=compressor)
    
    docs = compression_retriever.invoke(question)
    
    # Logic: Keep doc IF (target_vendors is Empty) OR (doc.vendor is IN target_vendors)
    final_docs = []
    for d in docs:
        doc_vendor = d.metadata.get("vendor_name")
        if target_vendors and doc_vendor not in target_vendors:
            continue # Skip doc if it belongs to a vendor we didn't ask for
        final_docs.append(d)
        
    return {"documents": final_docs}

def grade_documents(state):
    # Step 3: Documents ko Grade karo.
    print("---GRADE---")
    filtered = []
    for d in state["documents"]:
        score = grader_chain.invoke({"question": state["question"], "document": d.page_content})
        if score.binary_score == "yes":
            filtered.append(d)
    return {"documents": filtered}

def generate(state):
    """ Step 4: Final Answer Generate karo. """
    print("---GENERATE---")
    docs = state["documents"]
    
    if not docs:
        return {"generation": "I'm sorry, I couldn't find any relevant information in the documents."}
    
    # Check context for multiple vendors to adjust prompt
    vendors_in_context = set(d.metadata.get("vendor_name") for d in docs)
    is_comparison = len(vendors_in_context) > 1

    gen_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")
    
    if is_comparison:
        system_msg = "You are a Comparative Analysis Assistant. The user wants to compare multiple vendors. Create a Markdown Table comparing them based on the context. Summarize key differences."
    else:
        system_msg = "Answer strictly from context. Use Markdown tables for data representation where appropriate."

    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Question: {question}\nContext: {context}")
    ])
    chain = gen_prompt | gen_llm | StrOutputParser()
    
    # Prepare Context with Headers for better LLM understanding
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

# Add Nodes
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

# Edges
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