# Digital Contract Hub — Vibe Coding Plan
> OnPoint AI Innovation Engineer · Problem 2  
> Stack: PaddleOCR-VL 1.5 · LlamaIndex · ChromaDB · SQLite · Gemini API · Streamlit  
> Approach: Vertical slices — mỗi slice demo được độc lập

---

## Architecture tổng quan

```
┌──────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                      │
│                                                              │
│  PDF/Image                                                   │
│      ↓                                                       │
│  [Router] ──── text PDF ──→ pymupdf                          │
│      └──────── scanned  ──→ PaddleOCR-VL 1.5                │
│                                  ↓                           │
│                           [LLM Extractor]                    │
│                          /       |        \                  │
│                    ChromaDB   SQLite    BM25 Index           │
│                   (semantic) (struct)  (keyword)             │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                         │
│                                                              │
│  User Query → [Intent Router]                                │
│                   ↓           ↓          ↓                   │
│              Semantic     Structured   Keyword               │
│             (ChromaDB)    (Text-SQL)   (BM25)                │
│                   └───────────┴──────────┘                   │
│                              ↓                               │
│                     RRF Fusion + Rerank                      │
│                              ↓                               │
│                    Gemini → Answer + Citation                │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                       STREAMLIT UI                           │
│          Upload · Search · Browse · Clause Explorer          │
└──────────────────────────────────────────────────────────────┘
```

---

## Cấu trúc thư mục

```
contract-hub/
├── README.md
├── .env.example               # GEMINI_API_KEY
├── requirements.txt
│
├── ingestion/
│   ├── router.py              # text vs scanned detection
│   ├── parser.py              # pymupdf → text + page numbers
│   ├── ocr.py                 # PaddleOCR-VL 1.5 wrapper
│   ├── chunker.py             # clause-aware chunking
│   └── extractor.py           # LLM → structured JSON
│
├── indexing/
│   ├── vector_store.py        # ChromaDB
│   ├── bm25_store.py          # rank_bm25
│   └── sql_store.py           # SQLite + SQLAlchemy
│
├── retrieval/
│   ├── hybrid_search.py       # Vector + BM25 + RRF fusion
│   ├── reranker.py            # cross-encoder reranking
│   ├── text_to_sql.py         # structured query via Gemini
│   └── router.py              # intent classifier
│
├── generation/
│   ├── prompts.py             # tất cả prompt templates
│   └── answer.py              # Gemini call + citation format
│
├── ui/
│   └── app.py                 # Streamlit
│
├── eval/
│   ├── test_cases.json        # 10-15 known-answer queries
│   └── evaluate.py            # precision@3, recall, citation acc
│
└── data/
    ├── raw/                   # contract gốc
    ├── processed/             # JSON sau parse
    └── index/                 # ChromaDB persistent
```

---

## Vertical Slices — thứ tự vibe

### 🟢 Slice 1 — "1 contract, search được, có citation"
> Dừng ở đây cũng có thứ demo. Target: ~12h

**Bước 1 — Setup**
```bash
pip install pymupdf paddleocr paddlepaddle-gpu chromadb \
            llama-index rank_bm25 openai streamlit \
            sentence-transformers sqlalchemy
```

**.env**
```
GEMINI_API_KEY=...
```

**Bước 2 — PDF Router** `ingestion/router.py`
```python
# Nếu extract được >= 100 chars/page → text PDF
# Ngược lại → scanned
def route(pdf_path: str) -> Literal["text", "scanned"]:
    doc = fitz.open(pdf_path)
    total_chars = sum(len(p.get_text()) for p in doc)
    avg = total_chars / len(doc)
    return "text" if avg >= 100 else "scanned"
```

**Bước 3 — Parser** `ingestion/parser.py`
- pymupdf extract text + page number
- Detect bảng → markdown table
- Output: `List[{"text": str, "page": int, "type": str}]`

**Bước 4 — OCR** `ingestion/ocr.py`
```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    vl_rec_model_kwargs={"attn_implementation": "flash_attention_2"}
)
# ⚠️ Unload sau khi xong để free 6GB VRAM
```

**Bước 5 — Clause-aware Chunker** `ingestion/chunker.py`

Key insight: chunk theo điều khoản, không fixed-size
```
Regex patterns:
  - Tiếng Việt: r"^(Điều|ĐIỀU)\s+\d+"
  - English:    r"^(Article|Section|Clause)\s+\d+"
  - Số thứ tự:  r"^\d+\.\s+"

Nếu clause > 1000 tokens → sliding window 512/128 overlap
Mỗi chunk metadata: {contract_id, clause_number, page_start, page_end, clause_type}
```

> Đây là lý do để đạt citation đúng clause + page theo yêu cầu đề bài

**Bước 6 — LLM Structured Extractor** `ingestion/extractor.py`

Dùng Gemini model:
```json
{
  "contract_id": "HĐ-2024-001",
  "title": "Hợp đồng cung cấp dịch vụ IT",
  "parties": [
    {"name": "Công ty A", "role": "bên_a"},
    {"name": "Công ty B", "role": "bên_b"}
  ],
  "effective_date": "2024-01-01",
  "expiry_date": "2025-01-01",
  "contract_value": 500000000,
  "currency": "VND",
  "governing_law": "Việt Nam",
  "clauses": [
    {
      "clause_number": "Điều 5",
      "clause_type": "payment_terms",
      "page": 3,
      "summary": "Thanh toán trong 30 ngày"
    }
  ]
}
```

**Bước 7 — Indexing** 

ChromaDB (semantic):
```python
# Collection lưu chunks + metadata đầy đủ
collection.add(
    documents=[chunk.text],
    metadatas=[{"contract_id": ..., "clause_number": ..., "page": ...}],
    ids=[chunk.id]
)
```

BM25 (keyword):
```python
# Pickle lưu local, rebuild khi thêm contract mới
bm25 = BM25Okapi([chunk.text.split() for chunk in chunks])
```

SQLite (structured):
```sql
CREATE TABLE contracts (
    id TEXT PRIMARY KEY,
    title TEXT,
    value REAL,
    currency TEXT,
    effective_date DATE,
    expiry_date DATE,
    governing_law TEXT
);
CREATE TABLE parties (id INTEGER PRIMARY KEY, contract_id TEXT, name TEXT, role TEXT);
CREATE TABLE clauses (id INTEGER PRIMARY KEY, contract_id TEXT, number TEXT,
                      type TEXT, page INTEGER, summary TEXT);
```

**Bước 8 — Hybrid Search** `retrieval/hybrid_search.py`

RRF fusion — đơn giản nhưng hiệu quả:
```python
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

def fuse(vector_hits, bm25_hits) -> List[ScoredChunk]:
    scores = defaultdict(float)
    for rank, hit in enumerate(vector_hits):
        scores[hit.id] += rrf_score(rank)
    for rank, hit in enumerate(bm25_hits):
        scores[hit.id] += rrf_score(rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

> Vector giỏi semantic ("điều khoản phạt vi phạm")  
> BM25 giỏi exact match ("Điều 8.2", "500.000.000 VND")  
> Kết hợp → precision@3 > 90%

**Bước 9 — Reranker** `retrieval/reranker.py`
```python
# Free, local, ~200MB
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# Rerank top-10 → lấy top-3
```

**Bước 10 — Citation Prompt** `generation/prompts.py`
```
System: Bạn là trợ lý phân tích hợp đồng pháp lý.
        Chỉ trả lời dựa trên context được cung cấp.
        Mỗi claim PHẢI cite [Điều X, trang Y, Hợp đồng Z].
        Không tìm thấy → nói rõ "Không có trong tài liệu".

Context:
[Điều 8.2, trang 12, HĐ-2024-001]
Bên A có nghĩa vụ thanh toán trong vòng 30 ngày...

[Điều 9.1, trang 15, HĐ-2024-001]  
Phạt vi phạm tiến độ: 0.05%/ngày, tối đa 10%...

User: {query}
```

**Bước 11 — Streamlit MVP** `ui/app.py`
- Tab 1: Upload PDF → chạy pipeline → show progress bar
- Tab 2: Search box → answer + citations

---

### 🟡 Slice 2 — "Multi-contract + Smart Query Router"
> Nếu Slice 1 xong sớm. Target: +8h

**Intent Router** `retrieval/router.py`

Classify query trước khi search:
```python
INTENTS = {
    "semantic":    "Hỏi về nội dung, ý nghĩa điều khoản",
    "structured":  "Filter theo ngày, số tiền, tên công ty cụ thể",
    "keyword":     "Tìm điều khoản/số hợp đồng cụ thể",
}
# Gemini classify → route sang đúng retriever
```

**Text-to-SQL** `retrieval/text_to_sql.py`

Cho structured queries:
```
User: "Hợp đồng nào sắp hết hạn trong 30 ngày?"
→ SELECT * FROM contracts WHERE expiry_date <= date('now', '+30 days')

User: "Tổng giá trị hợp đồng với Công ty X?"
→ SELECT SUM(c.value) FROM contracts c JOIN parties p ON...
```

**UI mở rộng:**
- Contract Browser: list + filter theo party/date/value
- Clause Explorer: xem tất cả clauses của 1 contract theo type

---

### 🔵 Slice 3 — "Eval + Polish"
> Target: +6h

**Eval script** `eval/evaluate.py`

10-15 test cases:
```json
[
  {
    "query": "Điều khoản phạt vi phạm tiến độ?",
    "expected_clause": "Điều 9.1",
    "expected_page": 15,
    "expected_contains": ["0.05%", "10%"]
  }
]
```

Metrics tự động:
- **Precision@3**: retrieved top-3 có chứa expected_clause không
- **Citation accuracy**: answer có cite đúng clause + page không
- **Answer faithfulness** — LLM-as-judge

**UI polish:**
- Highlight source clause trong PDF viewer
- Export kết quả search ra CSV
- Dark mode 😄

---

### ⚪ Slice 4 — "Knowledge Graph" (nếu còn thời gian)
> Bonus. Target: +5h

Không dùng Neo4j (quá nặng setup), dùng **NetworkX** + visualize bằng **PyVis**:

```python
import networkx as nx

G = nx.DiGraph()
G.add_node("HĐ-2024-001", type="contract", value=500_000_000)
G.add_node("Công ty A", type="party")
G.add_edge("Công ty A", "HĐ-2024-001", relation="bên_a")
```

Multi-hop queries:
```python
# "Tất cả vendor có hợp đồng > 1 tỷ VND?"
vendors = [n for n, d in G.nodes(data=True) if d["type"] == "party"]
# traverse edges → filter contracts by value
```

Visualize trong Streamlit bằng `streamlit-agraph`.

> Mention trong README nếu không làm kịp:  
> *"Migrating to Neo4j would unlock multi-hop reasoning — e.g. overlapping vendors across contract periods"*

---

## Tech Stack

| Layer | Tool | Lý do |
|---|---|---|
| OCR scan | PaddleOCR-VL 1.5 + flash_attn2 | SOTA, free, tiếng Việt, 6GB OK |
| PDF text | pymupdf | Nhanh, chính xác 100% |
| LLM extract | Gemini | Structured JSON extraction |
| Embedding | text-embedding-3-small | $0.02/1M tokens |
| Vector DB | ChromaDB local | Zero setup |
| Keyword | rank_bm25 | Pure Python |
| Structured | SQLite + SQLAlchemy | Không cần server |
| Reranker | ms-marco-MiniLM-L-6-v2 | Free, local |
| RAG | LlamaIndex | Native citation |
| LLM answer | Gemini | Citation-following |
| UI | Streamlit | Nhanh nhất cho POC |
| KG (optional) | NetworkX + PyVis | Nhẹ, không cần Neo4j |

---

## Evaluation Criteria mapping

| Tiêu chí đề bài | Cách đạt |
|---|---|
| OCR > 99% | PaddleOCR-VL 1.5 (SOTA) + pymupdf cho text PDF |
| Precision@3 > 90% | Hybrid search RRF + cross-encoder rerank |
| Citation clause + page | Clause-aware chunking + citation prompt |
| Clause recall > 85% | LLM extractor với JSON schema cụ thể |

**Proposed thêm:**
- **Answer faithfulness** — LLM-as-judge, không hallucinate ngoài context
- **Ingestion throughput** — pages/minute, đo được và optimize được

---

## Limitations (viết thẳng trong README)

- OCR chậm ~20-30s/trang trên 6GB VRAM GPU — ổn cho batch ingestion, không real-time
- Ảnh chụp mờ/nghiêng > 30° giảm accuracy — preprocessing có thể giúp
- Font tiếng Việt cũ đôi khi bị lỗi dấu — cần post-processing normalization
- LLM hallucinate nếu context mơ hồ → hệ thống fallback "Không tìm thấy"
- SQLite không scale > 10K contracts — migrate Postgres khi production

---

## Prompt vibe coding gợi ý

Khi pair với AI, dùng prompts dạng:

```
"Viết ingestion/router.py, input: pdf_path string,
output: Literal['text','scanned'], dùng pymupdf,
threshold 100 chars/page, có docstring"

"Viết hybrid_search.py dùng RRF fusion,
input: query string + chroma collection + bm25 index,
output: List[ScoredChunk] top 10, k=60"

"Viết citation prompt template, system prompt enforce
cite [Điều X, trang Y, HĐ Z] cho mọi claim,
fallback 'Không có trong tài liệu' nếu không tìm thấy"
```

---

## README outline (viết cuối cùng)

```markdown
# Digital Contract Hub

## What it does (1 paragraph, non-technical)
## Quick Start (3 commands)
## Architecture (paste sơ đồ ASCII ở trên)
## How to use
  - Upload contract
  - Search examples
## Limitations
## What I'd do with more time
  - Neo4j KG
  - Local LLM (Qwen2.5-7B) cho data privacy
  - Fine-tuned NER cho tiếng Việt
```
