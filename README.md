# Agents

### analyze_query:

Before user query ::
AVAILABLE_VENDORS : vector database me total kitne vendors hain already present hota h ingest krte waqt hi
AVAILABLE_SECTIONS : kitne sections hain

After user query ::
query : What is the pricing of Eltrix? -> LLM
LLM prompt -> Question se Vendors nikalo, Sections nikalo, aur question ko saaf (standalone) karke likho.
Pydantic -> Ye sabse powerful cheez hai. Ye LLM ko force karta hai ki wo kachra output na de, balki ek strict JSON format de jisme sirf 3 box hon: vendors, section, aur standalone_question.
Output: LLM ek raw guess marta hai. (Maano usne vendor nikala: ["Eltrixx"] aur section nikala: ["price"]).

Verification ::
LLM ka vendors extracted> ko string comparison krta h available vendors ke list me sbse difflib ka use krke agr 70% se jyada accuracy h to final vendor me add krta h
LLM ka section extracted> ko string comparion difflab wala aur 60% accuracy ke saath aur subset match krta h eg: agr price extract kara llm ne aur available vendor me pricing h to wo pricing ko include kr lega.

final output ::
After verifying everything it will return dictionary>
{"filters": {"vendors": ["Eltrix"], "sections": ["Pricing and Maintenance"]}, "question": "What is the pricing of Eltrix?"}

### Retrieve

prerequisite setup::
1. bm25_retriever : Keyword search krta h
2. Qdrant : database jha pe vector stored h
3. reranker(compressor) : question ke according top n document krega sb ka sb ni krega

user query::
pichle node se question, vendor_name aur sections_name nikalta h 

dense retriever::
child doc vector ke saath similarity search hota h with filter aur 20 chunks wapas aate h
Is 20 chunks ke parent documents laate h metadata ke doc_id ke help se aur parent_doc variable me store krte h

sparse retriever::
ye keyword search krta h sb document pe qki bm25 me filters apply ni hota qdrant ke ander
Manually pyhton se filter apply hota h retrieved document pe
aur saare document ek filtered_sparse_docs me store hota h

Deduplication::
kuch document aise bhi ho skte h jo filtered_sparse_docs aur parent_doc dono me ho
all_docs variable me saare unique document h sparse aur dense dono milake

Reranker::
reranker ko ham question aur all_docs dete h, reranker top5 documents jo question ko satify kr rha ho wo final_docs me store hote h

### Generate

Document check::
ye document check krta h agr koi document iske paas aaya hi ni h retrieve hokr to "sorry no relevance doc to this question" output me

Context building::
ye har chunks ko ek single string me jodta h aur har chunks ke upar uska vendor_name aur section likh deta h xml format me 

answer generation::
prompt context jo maine doc se nikala aur question teeno ko llm ke paas bhejta h aur final answer return krta h 


# Evaluation

### Pipeline

```
User Query
    ↓
Retriever
    ↓
Top-5 Documents
    ↓
Ground Truth Documents se comparison
    ↓
Precision@5, Recall@5, MRR@5, nDCG@5
```

### Example (SciFact)

**Query:**
> "Vitamin D deficiency is associated with increased risk of cardiovascular disease"

**SciFact Ground Truth**
Relevant Documents: `[Doc_10, Doc_25]`

**RAG Retriever Output**
Top-5 Retrieved: `[Doc_25, Doc_80, Doc_10, Doc_44, Doc_91]`

Ab evaluation metrics isi result ko different angles se measure karte hain.

---

## 1. Precision@5

Jo 5 documents tumne retrieve kiye, unmein se kitne actually relevant the?

**Ground Truth:** `[Doc_10, Doc_25]`
**Retrieved Top-5:** `[Doc_25, Doc_80, Doc_10, Doc_44, Doc_91]`

| Retrieved | Relevant? |
|:---------:|:---------:|
| Doc_25    | ✅ YES    |
| Doc_80    | ❌ NO     |
| Doc_10    | ✅ YES    |
| Doc_44    | ❌ NO     |
| Doc_91    | ❌ NO     |

- Relevant retrieved documents = 2
- Total retrieved documents = 5

**Precision@5 = 2/5 = 40%**

---

## 2. Recall@5

Ground Truth mein jitne relevant documents the, unmein se tumne top-5 mein kitne find kar liye?

> "Jo documents actually relevant the, unmein se main kitne retrieve kar paya?"

**Ground Truth:** `[Doc_10, Doc_25]`
**Retrieved Top-5:** `[Doc_25, Doc_80, Doc_10, Doc_44, Doc_91]`

- Ground truth mein total relevant documents = 2
- Retrieved relevant documents = Doc_25, Doc_10 = 2

**Recall@5 = 2/2 = 100%**

---

## 3. MRR@5 — Mean Reciprocal Rank

Tumhare retrieved documents mein pehla relevant document generally kitni upar rank par aa raha hai.

Agar MRR = 90% hai, matlab average mein first relevant document bahut high rank par aa raha hai.

**Ground Truth:** `[Doc_10, Doc_25]`

**Query 1**

| Rank | Document | Relevant? |
|:----:|:--------:|:---------:|
| 1    | Doc_80   | ❌        |
| 2    | Doc_90   | ❌        |
| 3    | Doc_25   | ✅        |
| 4    | Doc_50   | ❌        |
| 5    | Doc_70   | ❌        |

First relevant document Rank = 3
Reciprocal Rank = 1/3 = 0.3333

**Query 2**

| Rank | Document | Relevant? |
|:----:|:--------:|:---------:|
| 1    | Doc_25   | ✅        |
| 2    | Doc_90   | ❌        |
| 3    | Doc_80   | ❌        |
| 4    | Doc_50   | ❌        |
| 5    | Doc_70   | ❌        |

First relevant document Rank = 1
Reciprocal Rank = 1/1 = 1

**Query 3**

| Rank | Document | Relevant? |
|:----:|:--------:|:---------:|
| 1    | Doc_90   | ❌        |
| 2    | Doc_80   | ❌        |
| 3    | Doc_50   | ❌        |
| 4    | Doc_70   | ❌        |
| 5    | Doc_25   | ✅        |

First relevant document Rank = 5
Reciprocal Rank = 1/5 = 0.2

**MRR Calculation:**
```
MRR = (0.333 + 1 + 0.2) / 3 = 0.511
```

**MRR@5 = 51.1%**

---

## 4. nDCG@5 — Normalized Discounted Cumulative Gain

Tumhare retrieved documents ki ranking, ideal ranking ke comparison mein kitne % matching hai.

Agar nDCG = 90% hai, matlab tumhare retrieved documents ki ranking, ideal ranking ke comparison mein approximately 90% normalized matching hai.