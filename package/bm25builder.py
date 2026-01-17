from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_community.retrievers import BM25Retriever
import pickle

DOC_STORE_PATH = "./persistent_doc_store"

raw_store = LocalFileStore(DOC_STORE_PATH)
docstore = EncoderBackedStore(
    store=raw_store,
    key_encoder=lambda k: k,
    value_serializer=pickle.dumps,
    value_deserializer=pickle.loads
)

def build_bm25_retriever(docstore, k=20):
    bm25_docs = []

    try:
        keys = list(docstore.yield_keys())
        if keys:
            bm25_docs = [
                doc for doc in docstore.mget(keys)
                if doc is not None
            ]
    except Exception as e:
        print(f"Error loading disk store for BM25: {e}")

    if bm25_docs:
        retriever = BM25Retriever.from_documents(bm25_docs)
        retriever.k = k
    else:
        retriever = BM25Retriever.from_texts(
            ["Empty"], metadatas=[{}]
        )

    return retriever
