import json
import math
from tqdm import tqdm

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from package import docstore_metadata
from package import bm25builder


CLAIMS_PATH = "data/claims_dev.jsonl"
COLLECTION_NAME = "_RAG_DB_SCIFACT"
QDRANT_URL = "http://localhost:6333"
K_RETRIEVAL = 20
TOP_K_EVAL = 5

# 1. INITIALIZE RAG COMPONENTS
print("Initializing RAG Components...")
# Embedding model
embd = OllamaEmbeddings(    model="nomic-embed-text:latest")

# Qdrant
client = QdrantClient(    url=QDRANT_URL)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embd
)

# ------------------------------------------------------------
# Document store
# ------------------------------------------------------------
docstore = docstore_metadata.docstore

# ------------------------------------------------------------
# BM25 retriever
# ------------------------------------------------------------
bm25_retriever = bm25builder.build_bm25_retriever(
    docstore,
    k=K_RETRIEVAL
)

# ------------------------------------------------------------
# Cross encoder reranker
# ------------------------------------------------------------
reranker_model = HuggingFaceCrossEncoder(
    model_name="/home/ppc/models/bge-reranker-v2-m3"
)

compressor = CrossEncoderReranker(
    model=reranker_model,
    top_n=TOP_K_EVAL
)


# ============================================================
# 2. METRIC FUNCTIONS
# ============================================================

def get_dcg(rel_array):
    """
    Calculate Discounted Cumulative Gain.

    For binary relevance:

        relevant     -> 1
        non-relevant -> 0

    Formula:

        DCG = sum((2^rel - 1) / log2(rank + 1))

    Example:

        rel_array = [1, 1, 0, 0, 0]

    """

    dcg = 0.0

    for rank, rel in enumerate(rel_array, start=1):

        dcg += (
            (2 ** rel - 1)
            / math.log2(rank + 1)
        )

    return dcg


def calculate_precision_at_k(
    retrieved_ids,
    ground_truth_ids,
    k
):
    """
    Precision@K

        Precision@K =
            number of relevant retrieved documents
            -------------------------------------
                          K

    """

    retrieved_at_k = retrieved_ids[:k]

    ground_truth_set = set(ground_truth_ids)

    hits = len(
        set(retrieved_at_k) & ground_truth_set
    )

    return hits / k


def calculate_recall_at_k(
    retrieved_ids,
    ground_truth_ids,
    k
):
    """
    Recall@K

        Recall@K =
            relevant documents retrieved in top-K
            -------------------------------------
                total relevant documents
    """

    if not ground_truth_ids:
        return 0.0

    retrieved_at_k = retrieved_ids[:k]

    ground_truth_set = set(ground_truth_ids)

    hits = len(
        set(retrieved_at_k) & ground_truth_set
    )

    return hits / len(ground_truth_set)


def calculate_mrr_at_k(
    retrieved_ids,
    ground_truth_ids,
    k
):
    """
    Mean Reciprocal Rank@K for a single query.

    Only the FIRST relevant document matters.

    Example:

        Retrieved:
            [101, 205, 300, 500, 600]

        Relevant:
            [300]

        MRR@5 = 1 / 3
    """

    ground_truth_set = set(ground_truth_ids)

    for rank, doc_id in enumerate(
        retrieved_ids[:k],
        start=1
    ):

        if doc_id in ground_truth_set:

            return 1.0 / rank

    return 0.0


def calculate_ndcg_at_k(
    retrieved_ids,
    ground_truth_ids,
    k
):
    """
    Calculate binary nDCG@K.

    Relevance:

        relevant document     -> 1
        non-relevant document -> 0
    """

    ground_truth_set = set(ground_truth_ids)

    retrieved_at_k = retrieved_ids[:k]

    # --------------------------------------------------------
    # Actual relevance list
    # --------------------------------------------------------

    rel_array = [
        1 if doc_id in ground_truth_set else 0
        for doc_id in retrieved_at_k
    ]

    # --------------------------------------------------------
    # Pad retrieved list to K positions.
    #
    # This matters when the retriever returns fewer than K
    # documents.
    # --------------------------------------------------------

    if len(rel_array) < k:

        rel_array += [
            0
        ] * (
            k - len(rel_array)
        )

    # --------------------------------------------------------
    # Ideal ranking
    #
    # Put all relevant documents at the top.
    #
    # Maximum number of relevant documents possible in
    # top-K is K.
    # --------------------------------------------------------

    number_of_relevant_documents = min(
        len(ground_truth_set),
        k
    )

    ideal_rel_array = (
        [1] * number_of_relevant_documents
        +
        [0] * (
            k - number_of_relevant_documents
        )
    )

    # --------------------------------------------------------
    # DCG
    # --------------------------------------------------------

    dcg = get_dcg(rel_array)

    # --------------------------------------------------------
    # Ideal DCG
    # --------------------------------------------------------

    idcg = get_dcg(
        ideal_rel_array
    )

    # --------------------------------------------------------
    # Avoid division by zero
    # --------------------------------------------------------

    if idcg == 0:

        return 0.0

    return dcg / idcg


# ============================================================
# 3. CORE RETRIEVAL FUNCTION
# ============================================================

def retrieve_top_k(
    query,
    top_k=TOP_K_EVAL
):
    """
    Retrieval pipeline:

        Dense retrieval
            +
        Parent document lookup
            +
        BM25 retrieval
            +
        Deduplication
            +
        Cross-encoder reranking
            +
        Unique document IDs
    """

    # ========================================================
    # Dense Search
    # ========================================================

    child_docs = vectorstore.similarity_search(
        query,
        k=K_RETRIEVAL
    )

    # --------------------------------------------------------
    # Extract parent document IDs
    # --------------------------------------------------------

    parent_ids = []

    for doc in child_docs:

        doc_id = doc.metadata.get("doc_id")

        if doc_id is not None:

            parent_ids.append(
                str(doc_id)
            )

    # --------------------------------------------------------
    # Remove duplicate parent IDs while preserving order.
    #
    # DO NOT use set() here because set() does not preserve
    # retrieval ranking order.
    # --------------------------------------------------------

    unique_parent_ids = list(
        dict.fromkeys(parent_ids)
    )

    # --------------------------------------------------------
    # Retrieve parent documents
    # --------------------------------------------------------

    parent_docs = [
        doc
        for doc in docstore.mget(
            unique_parent_ids
        )
        if doc is not None
    ]

    # ========================================================
    # Sparse Search
    # ========================================================

    sparse_docs = bm25_retriever.invoke(
        query
    )

    # ========================================================
    # Combine Dense + Sparse
    # ========================================================

    # Deduplicate by document content
    unique_docs_map = {}

    for doc in (
        parent_docs + sparse_docs
    ):

        unique_docs_map[
            doc.page_content
        ] = doc

    all_docs = list(
        unique_docs_map.values()
    )

    # ========================================================
    # Reranking
    # ========================================================

    if not all_docs:

        return []

    final_docs = compressor.compress_documents(
        documents=all_docs,
        query=query
    )

    # ========================================================
    # Extract UNIQUE document IDs
    #
    # Preserve reranker ordering.
    # ========================================================

    result_ids = []

    seen_ids = set()

    for doc in final_docs:

        doc_id = doc.metadata.get(
            "doc_id"
        )

        if doc_id is None:
            continue

        doc_id = str(doc_id)

        if doc_id in seen_ids:
            continue

        result_ids.append(
            doc_id
        )

        seen_ids.add(
            doc_id
        )

        if len(result_ids) >= top_k:

            break

    return result_ids


# ============================================================
# 4. RUN EVALUATION
# ============================================================

print(
    "\nStarting Evaluation on SciFact Dev Set..."
)


# ------------------------------------------------------------
# Accumulators
# ------------------------------------------------------------

precision_sum = 0.0
recall_sum = 0.0
mrr_sum = 0.0
ndcg_sum = 0.0

total_queries = 0


# ============================================================
# Read SciFact claims
# ============================================================

with open(
    CLAIMS_PATH,
    "r",
    encoding="utf-8"
) as f:

    lines = f.readlines()


# ============================================================
# Evaluate each query
# ============================================================

for line in tqdm(
    lines,
    desc="Evaluating Queries"
):

    data = json.loads(line)

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    query = data["claim"]

    # --------------------------------------------------------
    # Ground-truth evidence
    #
    # SciFact structure:
    #
    # "evidence": {
    #     "123": [...],
    #     "456": [...]
    # }
    #
    # The keys are the relevant document IDs.
    # --------------------------------------------------------

    evidence = data.get(
        "evidence",
        {}
    )

    ground_truth_ids = list(
        dict.fromkeys(
            str(doc_id)
            for doc_id in evidence.keys()
        )
    )

    # --------------------------------------------------------
    # Skip queries without ground truth
    # --------------------------------------------------------

    if not ground_truth_ids:

        continue

    total_queries += 1

    # ========================================================
    # Retrieval
    # ========================================================

    retrieved_ids = retrieve_top_k(
        query,
        top_k=TOP_K_EVAL
    )

    # Ensure IDs are strings
    retrieved_ids = [
        str(doc_id)
        for doc_id in retrieved_ids
        if doc_id is not None
    ]

    # --------------------------------------------------------
    # Safety deduplication.
    #
    # retrieve_top_k already removes duplicates, but doing
    # this here guarantees the evaluation is document-level.
    # --------------------------------------------------------

    unique_retrieved_ids = []

    seen_ids = set()

    for doc_id in retrieved_ids:

        if doc_id in seen_ids:
            continue

        unique_retrieved_ids.append(
            doc_id
        )

        seen_ids.add(
            doc_id
        )

    retrieved_ids = unique_retrieved_ids[
        :TOP_K_EVAL
    ]

    # ========================================================
    # Calculate Precision@K
    # ========================================================

    query_precision = calculate_precision_at_k(
        retrieved_ids=retrieved_ids,
        ground_truth_ids=ground_truth_ids,
        k=TOP_K_EVAL
    )

    # ========================================================
    # Calculate Recall@K
    # ========================================================

    query_recall = calculate_recall_at_k(
        retrieved_ids=retrieved_ids,
        ground_truth_ids=ground_truth_ids,
        k=TOP_K_EVAL
    )

    # ========================================================
    # Calculate MRR@K
    # ========================================================

    query_mrr = calculate_mrr_at_k(
        retrieved_ids=retrieved_ids,
        ground_truth_ids=ground_truth_ids,
        k=TOP_K_EVAL
    )

    # ========================================================
    # Calculate nDCG@K
    # ========================================================

    query_ndcg = calculate_ndcg_at_k(
        retrieved_ids=retrieved_ids,
        ground_truth_ids=ground_truth_ids,
        k=TOP_K_EVAL
    )

    # ========================================================
    # Accumulate
    # ========================================================

    precision_sum += query_precision
    recall_sum += query_recall
    mrr_sum += query_mrr
    ndcg_sum += query_ndcg


# ============================================================
# 5. FINAL RESULTS
# ============================================================

print(
    "\n" + "=" * 50
)

print(
    f"📊 EVALUATION RESULTS @ {TOP_K_EVAL}"
)

print(
    "=" * 50
)

print(
    f"Total Evaluated Queries: {total_queries}"
)


# ============================================================
# Prevent division by zero
# ============================================================

if total_queries == 0:

    print(
        "No queries with ground-truth evidence were evaluated."
    )

else:

    # --------------------------------------------------------
    # Macro-average
    #
    # Every query contributes equally.
    # --------------------------------------------------------

    precision_at_k = (
        precision_sum
        / total_queries
    )

    recall_at_k = (
        recall_sum
        / total_queries
    )

    mrr_at_k = (
        mrr_sum
        / total_queries
    )

    ndcg_at_k = (
        ndcg_sum
        / total_queries
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"Precision@{TOP_K_EVAL}: {precision_at_k:.4f}"
    )

    print(
        f"Recall@{TOP_K_EVAL}:    {recall_at_k:.4f}"
    )

    print(
        f"MRR@{TOP_K_EVAL}:       {mrr_at_k:.4f}"
    )

    print(
        f"nDCG@{TOP_K_EVAL}:      {ndcg_at_k:.4f}"
    )


print(
    "=" * 50
)