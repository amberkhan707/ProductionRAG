import json
import math
import csv
from tqdm import tqdm

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from package import docstore_metadata
from package import bm25builder

# ============================================================
# NEW BEIR CONFIGURATION FOR FIQA
# ============================================================
QUERIES_PATH = "fiqa_data/queries.jsonl"
QRELS_PATH = "fiqa_data/qrels/dev.tsv"  # Aap test.tsv bhi use kar sakte hain
COLLECTION_NAME = "FIQA_RAG_DB"
QDRANT_URL = "http://localhost:6333"
K_RETRIEVAL = 20
TOP_K_EVAL = 3

# 1. INITIALIZE RAG COMPONENTS
print("Initializing RAG Components...")
embd = OllamaEmbeddings(model="nomic-embed-text:latest")
client = QdrantClient(url=QDRANT_URL)
vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embd)

docstore = docstore_metadata.docstore
bm25_retriever = bm25builder.build_bm25_retriever(docstore, k=K_RETRIEVAL)

reranker_model = HuggingFaceCrossEncoder(model_name="/home/ppc/models/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=reranker_model, top_n=TOP_K_EVAL)

# 2. METRIC FUNCTIONS
def get_dcg(rel_array):
    dcg = 0.0
    for rank, rel in enumerate(rel_array, start=1):
        dcg += ((2 ** rel - 1) / math.log2(rank + 1))
    return dcg

def calculate_precision_at_k(retrieved_ids, ground_truth_ids, k):
    retrieved_at_k = retrieved_ids[:k]
    ground_truth_set = set(ground_truth_ids)
    hits = len(set(retrieved_at_k) & ground_truth_set)
    return hits / k

def calculate_recall_at_k(retrieved_ids, ground_truth_ids, k):
    if not ground_truth_ids:
        return 0.0
    retrieved_at_k = retrieved_ids[:k]
    ground_truth_set = set(ground_truth_ids)
    hits = len(set(retrieved_at_k) & ground_truth_set)
    return hits / len(ground_truth_set)

def calculate_mrr_at_k(retrieved_ids, ground_truth_ids, k):
    ground_truth_set = set(ground_truth_ids)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in ground_truth_set:
            return 1.0 / rank
    return 0.0

def calculate_ndcg_at_k(retrieved_ids, ground_truth_ids, k):
    ground_truth_set = set(ground_truth_ids)
    retrieved_at_k = retrieved_ids[:k]
    rel_array = [1 if doc_id in ground_truth_set else 0 for doc_id in retrieved_at_k]

    if len(rel_array) < k:
        rel_array += [0] * (k - len(rel_array))

    number_of_relevant_documents = min(len(ground_truth_set), k)
    ideal_rel_array = ([1] * number_of_relevant_documents + [0] * (k - number_of_relevant_documents))

    dcg = get_dcg(rel_array)
    idcg = get_dcg(ideal_rel_array)

    if idcg == 0:
        return 0.0
    return dcg / idcg

# 3. CORE RETRIEVAL FUNCTION
def retrieve_top_k(query, top_k=TOP_K_EVAL):
    child_docs = vectorstore.similarity_search(query, k=K_RETRIEVAL)

    parent_ids = []
    for doc in child_docs:
        doc_id = doc.metadata.get("doc_id")
        if doc_id is not None:
            parent_ids.append(str(doc_id))

    unique_parent_ids = list(dict.fromkeys(parent_ids))
    parent_docs = [doc for doc in docstore.mget(unique_parent_ids) if doc is not None]
    
    sparse_docs = bm25_retriever.invoke(query)

    unique_docs_map = {}
    for doc in (parent_docs + sparse_docs):
        unique_docs_map[doc.page_content] = doc

    all_docs = list(unique_docs_map.values())

    if not all_docs:
        return []

    final_docs = compressor.compress_documents(documents=all_docs, query=query)

    result_ids = []
    seen_ids = set()

    for doc in final_docs:
        doc_id = doc.metadata.get("doc_id")
        if doc_id is None:
            continue
        
        doc_id = str(doc_id)
        if doc_id in seen_ids:
            continue

        result_ids.append(doc_id)
        seen_ids.add(doc_id)

        if len(result_ids) >= top_k:
            break

    return result_ids

# ============================================================
# 4. RUN EVALUATION (UPDATED FOR BEIR FORMAT)
# ============================================================
print("\nLoading FIQA Ground Truth (qrels)...")
qrels = {}
with open(QRELS_PATH, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader) # Skip TSV header (query-id, corpus-id, score)
    for row in reader:
        if len(row) >= 3:
            q_id, doc_id, score = row[0], row[1], int(row[2])
            if score > 0: # Keep only relevant docs
                if q_id not in qrels:
                    qrels[q_id] = []
                qrels[q_id].append(str(doc_id))

print("Loading FIQA Queries...")
queries_dict = {}
with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        queries_dict[str(data["_id"])] = data["text"]

print("\nStarting Evaluation on FIQA Set...")

precision_sum = 0.0
recall_sum = 0.0
mrr_sum = 0.0
ndcg_sum = 0.0
total_queries = 0

# Loop through qrels (only evaluate queries that have ground truth)
for q_id, ground_truth_ids in tqdm(qrels.items(), desc="Evaluating Queries"):
    query = queries_dict.get(q_id)
    
    if not query or not ground_truth_ids:
        continue

    total_queries += 1
    
    # Run Retrieval
    retrieved_ids = retrieve_top_k(query, top_k=TOP_K_EVAL)

    # Ensure IDs are strings and deduplicated safely
    retrieved_ids = [str(doc_id) for doc_id in retrieved_ids if doc_id is not None]
    unique_retrieved_ids = list(dict.fromkeys(retrieved_ids))[:TOP_K_EVAL]

    # Calculate Metrics
    precision_sum += calculate_precision_at_k(unique_retrieved_ids, ground_truth_ids, TOP_K_EVAL)
    recall_sum += calculate_recall_at_k(unique_retrieved_ids, ground_truth_ids, TOP_K_EVAL)
    mrr_sum += calculate_mrr_at_k(unique_retrieved_ids, ground_truth_ids, TOP_K_EVAL)
    ndcg_sum += calculate_ndcg_at_k(unique_retrieved_ids, ground_truth_ids, TOP_K_EVAL)

# 5. FINAL RESULTS
print("\n" + "=" * 50)
print(f"📊 EVALUATION RESULTS @ {TOP_K_EVAL} for FIQA")
print("=" * 50)
print(f"Total Evaluated Queries: {total_queries}")

if total_queries == 0:
    print("No queries with ground-truth evidence were evaluated.")
else:
    print(f"Precision@{TOP_K_EVAL}: {precision_sum / total_queries:.4f}")
    print(f"Recall@{TOP_K_EVAL}:    {recall_sum / total_queries:.4f}")
    print(f"MRR@{TOP_K_EVAL}:       {mrr_sum / total_queries:.4f}")
    print(f"nDCG@{TOP_K_EVAL}:      {ndcg_sum / total_queries:.4f}")

print("=" * 50)