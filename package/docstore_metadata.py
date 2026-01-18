import pickle
from langchain_classic.storage import LocalFileStore, EncoderBackedStore

AVAILABLE_VENDORS = set()
AVAILABLE_SECTIONS = set()
DOC_STORE_PATH = "./persistent_doc_store"

raw_store = LocalFileStore(DOC_STORE_PATH)
docstore = EncoderBackedStore(
    store=raw_store,
    key_encoder=lambda k: k,
    value_serializer=pickle.dumps,
    value_deserializer=pickle.loads
)


def load_metadata_cache():
    AVAILABLE_VENDORS.clear()
    AVAILABLE_SECTIONS.clear()
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
                    raw_section_str = doc.metadata["section"]
                    if raw_section_str and isinstance(raw_section_str, str):
                        parts = raw_section_str.split(",")
                        for p in parts:
                            clean_sec = p.strip()
                            if clean_sec:
                                AVAILABLE_SECTIONS.add(clean_sec)
        
    except Exception as e:
        print(f"Error loading cache: {e}")

v_str = ", ".join(AVAILABLE_VENDORS) if AVAILABLE_VENDORS else "None"
s_str = ", ".join(AVAILABLE_SECTIONS) if AVAILABLE_SECTIONS else "None"
        