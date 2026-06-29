# FashionSearch AI

### A Generative AI-Powered Hybrid Retrieval System for Context-Aware Fashion Recommendations

---

## Overview

FashionSearch AI is an end-to-end intelligent shopping assistant that goes beyond traditional keyword search. It understands natural language fashion queries, extracts intent automatically, retrieves relevant products using a hybrid semantic + lexical pipeline, re-ranks them for precision, and generates explainable recommendations using a Large Language Model (LLM).

> **Example queries the system understands:**
> - *"pastel cotton kurta for summer under ₹2000"*
> - *"something festive and budget-friendly"*
> - *"navy blue office wear kurta"*

---

## Problem Statement

Online fashion marketplaces host thousands of products, but traditional keyword-based search systems fail users who express intent in natural language. They miss contextual queries, misinterpret style preferences, and provide zero explainability. FashionSearch AI solves this with a multi-stage AI pipeline.

---

## Architecture

```
User Query (Natural Language)
        ↓
[Query Understanding Layer]
  → Price / Brand / Colour / Category extraction (Regex + Heuristic NLU)
        ↓
[Hybrid Retrieval Layer]
  → BM25 Lexical Search (k=20)
  → FAISS Semantic Search (k=20)
        ↓
[Fusion & Filtering Layer]
  → Reciprocal Rank Fusion (RRF, k=60)
  → Metadata Filtering (price, category, brand, colour)
        ↓
[Precision Re-Ranking Layer]
  → Cross-Encoder (ms-marco-MiniLM-L-6-v2)
  → Rating-Aware Quality Boost (α = 0.3)
        ↓
[LLM Recommendation Layer]
  → GPT-3.5-turbo — context-grounded explanation
        ↓
[Gradio UI]
  → Product cards with images, prices, ratings, AI explanation
```

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Pipeline Orchestration | LangChain | Modular RAG architecture |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) | CPU-efficient semantic embeddings |
| Vector Store | FAISS (IndexFlatL2) | Exact nearest-neighbor semantic search |
| Lexical Retrieval | BM25 (rank_bm25) | Keyword-based matching |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) | Score-agnostic rank combination |
| Query Understanding | Regex + Heuristic NLU | Zero-latency intent extraction |
| Re-Ranking | `ms-marco-MiniLM-L-6-v2` + Rating Boost | Pairwise relevance + quality scoring |
| Generative Layer | OpenAI GPT-3.5-turbo | Natural language recommendation summaries |
| User Interface | Gradio | Interactive shopping assistant UI |
| Data Processing | Pandas, NumPy | Data cleaning and feature engineering |

---

## Dataset

- **Source:** Myntra Fashion Products Dataset
- **Size:** 4,000 products
- **Fields:** Product ID, Name, Category, Price, Colour, Brand, Image URL, Rating Count, Average Rating, Description, Product Attributes
- **Price Range:** ₹199 – ₹10,950

---

## Project Structure

```
Fashion_AI/
├── Fashionsearch_AI_Deepa.ipynb    # Main notebook — full implementation
├── README.md                       # Project documentation
└── FashionSearch_AI_Deepa_Aggarwal.docx  # Project report document
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local environment with CPU

### Install Dependencies

```bash
pip install langchain langchain-core langchain-community langchain-huggingface \
    faiss-cpu sentence-transformers rank_bm25 openai pandas numpy tqdm gradio
```

### Dataset Setup

Upload the Myntra Fashion Dataset CSV to Google Drive at:
```
/content/drive/MyDrive/FashionDataset/FashionDatasetv2.csv
```

### API Key (Optional — for LLM recommendations)

Set your OpenAI API key as an environment variable or in Google Colab secrets:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

> The system includes a graceful fallback if no API key is provided — it returns structured product results without AI-generated explanations.

---

## How to Run

1. Open `Fashionsearch_AI_Deepa.ipynb` in Google Colab
2. Mount Google Drive and verify the dataset path
3. Run all cells sequentially (Sections 1–9)
4. The Gradio UI launches automatically at the end with a public shareable link

---

## Pipeline Stages Explained

### 1. Data Preprocessing
- HTML tag stripping from product descriptions
- Text normalization (lowercasing, whitespace cleanup)
- Price segment labeling (budget / mid-range / premium)
- Schema validation and duplicate removal

### 2. Embedding Generation
- Each product's `name + category + colour + description` is concatenated into a rich text representation
- Embedded using `all-MiniLM-L6-v2` in batches of 32
- 384-dimensional vectors stored in FAISS `IndexFlatL2`

### 3. BM25 Lexical Index
- Tokenized product text corpus indexed with `rank_bm25`
- Captures exact keyword matches (brand names, product types)

### 4. Hybrid Retrieval with RRF
- Both BM25 and FAISS retrieve top-20 candidates independently
- Reciprocal Rank Fusion merges rankings: `score = Σ 1/(k + rank)` where k=60
- Semantic channel weighted at 0.6, lexical at 0.4

### 5. Metadata Filtering
- Regex extracts price ceiling, brand name, colour, and category from query
- Post-fusion filter eliminates non-matching products

### 6. Cross-Encoder Re-Ranking
- `ms-marco-MiniLM-L-6-v2` scores each (query, product) pair
- Final score: `CE_score + 0.3 × normalized_rating`
- Top-5 results selected

### 7. LLM Recommendation
- GPT-3.5-turbo receives structured product details as context
- Generates a natural language explanation of why each product matches the query

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `all-MiniLM-L6-v2` over `all-mpnet-base-v2` | ~3× faster on CPU with <2% quality difference |
| `IndexFlatL2` over HNSW/IVF | Exact search completes in <50ms for 4K products |
| RRF over weighted score averaging | Score-agnostic — avoids BM25 vs L2 normalization mismatch |
| Regex NLU over LLM-based parsing | Zero latency vs ~500ms overhead per query |
| Rating boost (α=0.3) | Prevents low-quality items from dominating top results |

---

## Performance Metrics

| Metric | Result |
|---|---|
| Products Indexed | 4,000 |
| Embedding Dimension | 384 |
| Retrieval + Re-Rank Latency | ~1.5–2.5s (CPU) |
| End-to-End with LLM | ~3–4s |
| Precision@5 | Strong alignment with query intent |
| MRR | First relevant result typically at rank 1 |

---

## Future Enhancements

- Multimodal search using image embeddings (CLIP)
- Conversational memory for session-aware recommendations
- Personalization via user interaction signals
- Approximate FAISS indexing (HNSW/IVF) for larger catalogs
- LLM-based query expansion and paraphrasing
- Fine-tuned domain-specific embeddings
- Multilingual support
- Streaming responses for improved perceived latency
- Production API deployment with caching layer

---

## Author

**Deepa Aggarwal**
FashionSearch AI — Generative AI Assignment
