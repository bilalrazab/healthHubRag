<div align="center">

# HealthHub RAG

**A production-grade multi-branch healthcare RAG system**
Built for a 12-branch Dubai clinic network · Hybrid SQL + Vector retrieval · WhatsApp-ready

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-00D4B8?style=flat-square)](https://www.trychroma.com)
[![Claude](https://img.shields.io/badge/Claude-Sonnet_4.6-D97757?style=flat-square)](https://anthropic.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com)
[![Render](https://img.shields.io/badge/Render-deployed-46E3B7?style=flat-square&logo=render&logoColor=white)](https://render.com)

[**Live Demo**](https://healthhub-rag.onrender.com) · [**Interactive Showcase**](https://bilalrazab.github.io/healthHubRag/SHOWCASE.html) · [**Architecture**](#architecture) · [**Setup**](#setup)

</div>

---

## What this is

HealthHub by Al-Futtaim runs **12 clinics across Dubai** — 152+ doctors, 25+ specialities, 18+ insurance providers. This project builds an AI patient assistant that answers real questions about that network: *"Is there an ENT doctor at Al Qusais who takes NAS insurance?"* — correctly, every time, grounded in real data.

It is **not** a LangChain wrapper around an LLM. Every layer — scraping, parsing, SQL retrieval, vector retrieval, keyword retrieval, fusion, intent classification, routing — is written from scratch. The point was to understand and own the full RAG pipeline, not assemble pre-built blocks.

```
Patient: "I have NAS insurance, is it covered in Qusais?"
Bot:     "✅ NAS insurance is accepted at HealthHub – Al Qusais.
          I'd recommend confirming when you book, as coverage details
          can sometimes vary. You can book via..."

Patient: "what about DFC?"
Bot:     "✅ NAS insurance is also accepted at HealthHub – Festival City (DFC)..."
```
*(Second query resolves "it" → NAS and "DFC" → Festival City automatically, from conversation history — no entity restated.)*

---

## Why it exists

Most "RAG chatbot" portfolio projects retrieve from a flat vector store and hope the LLM sorts it out. That breaks immediately at this scale — a semantic search for *"ENT doctor"* will happily return a General Practitioner's profile if it scored slightly higher, because vector similarity doesn't know about your business logic. **This project exists to solve that properly**: structured facts (branch hours, doctor-to-branch mappings, insurance coverage) live in SQL where they belong; only genuinely semantic content (speciality descriptions, doctor bios, health articles) lives in the vector store — and even there, every chunk carries metadata filters so retrieval is scoped *before* ranking, not after.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  DATA LAYER (one-time pipeline)                                      │
│                                                                        │
│   Firecrawl ──► parser.py ──► structured_data.json                   │
│  (JS render)    (regex/NLP    (single source of truth)               │
│                  extraction)         │                               │
│                                       ├──► db_loader.py ──► SQLite    │
│                                       │    (12 tables, FK joins)      │
│                                       │                               │
│                                       └──► vec_loader.py ──► ChromaDB │
│                                            (paragraph chunking,       │
│                                             rich metadata tags)       │
└──────────────────────────────────────────────────────────────────────┘
                                  │
┌──────────────────────────────────────────────────────────────────────┐
│  QUERY LAYER (per message)                                            │
│                                                                        │
│   guards.py ──► intent.py ──► router.py ──► [SQL | Vector | BM25]    │
│   (emergency/    (Claude classifies      (per-intent retrieval        │
│    complaint,     + extracts branch/      strategy — see table        │
│    0 API cost)    speciality/doctor/      below)                      │
│                   insurance entities,            │                    │
│                   uses conv. history)            ▼                    │
│                                            rrf.py (fuses vector        │
│                                             + BM25 when both used)     │
│                                                   │                    │
│                                                   ▼                    │
│                                       context_builder.py               │
│                                       (SQL facts first, then           │
│                                        vector context, capped)         │
│                                                   │                    │
│                                                   ▼                    │
│                                       Claude Sonnet 4.6 (grounded      │
│                                        response, never hallucinates    │
│                                        beyond retrieved context)       │
└──────────────────────────────────────────────────────────────────────┘
                                  │
┌──────────────────────────────────────────────────────────────────────┐
│  TRANSPORT (same handle_message() function, three faces)              │
│   interfaces/cli.py  →  Terminal REPL with live pipeline trace        │
│   interfaces/api.py  →  FastAPI + web chat UI (Render-deployed)       │
│   WhatsApp (planned) →  drop-in webhook, zero RAG logic changes       │
└──────────────────────────────────────────────────────────────────────┘
```

### Intent → retrieval routing

The router is the core design decision. Every intent gets exactly the retrieval strategy it needs — no wasted vector searches on factual lookups, no SQL queries for genuinely open-ended questions.

| Intent | Route | Why |
|---|---|---|
| `branch_info` / `branch_hours` | SQL only | Address, phone, hours are facts in a table, not prose to search |
| `doctor_search` | SQL primary, vector fallback | Branch+speciality+name filters resolve via JOIN; vector only fires if SQL returns nothing |
| `insurance_check` | SQL only | Exact branch×insurer lookup — no ambiguity to resolve semantically |
| `speciality_info` | SQL (which branches) + Vector (filtered to `speciality` chunks) | Hybrid — needs both the fact and the description |
| `general_health` | Vector + BM25 → RRF fusion | Genuinely open-ended, needs both semantic recall and keyword precision |
| `emergency` / `complaint` | Hardcoded rule | **Zero LLM calls.** Safety-critical paths never depend on classification |

---

## What's actually hard about this (and how it's solved)

**The site has zero static content.** HealthHub's site is Elementor/WordPress with everything client-rendered — `httpx` + BeautifulSoup returned empty shells on every single page. Switched the entire scraper to **Firecrawl**, which runs a real headless browser and waits for JS to settle before returning markdown.

**Doctor data lives in two places with inconsistent formatting.** Branch pages list mini doctor "stubs" inline; full profiles live on separate `/doctor/` pages. `parser.py` reconciles both sources, deduplicates by slug, and falls back to stub data when a profile page failed to scrape — so doctor count never silently drops.

**Branch names don't match between data sources.** The branches table uses `"HealthHub – Al Karama"` (en-dash); doctor profile pages list clinics as `"HealthHub - Al Karama"` (hyphen). `db_loader.py` normalises both before any join, plus maintains an explicit alias map for abbreviations patients actually type (`DFC`, `JVC`, `DSO`).

**Real conversations have pronouns and ellipsis.** *"is it covered in DFC?"* has no insurance name in it — the patient already said NAS three turns ago. `intent.py` passes the last 3 conversation turns into the classification prompt itself, so entity extraction resolves "it" correctly instead of treating each message in isolation.

**Free-tier hosting has 512MB RAM.** `sentence-transformers` + `torch` alone is ~400MB resident. Swapped to ChromaDB's `DefaultEmbeddingFunction` (same `all-MiniLM-L6-v2` model, served via `onnxruntime`) — same retrieval quality, fraction of the memory footprint, runs comfortably on Render's free tier.

---

## Data model

Twelve scraped fields aren't enough to run a realistic clinic assistant — there's no public API for doctor schedules or per-branch insurance networks. Rather than fake the whole system, the architecture **clearly separates real from simulated data** at the schema level:

```sql
-- REAL — scraped from healthhubalfuttaim.com
branches, doctors, specialities, branch_specialities,
insurance_providers, health_packages

-- SIMULATED — generated, flagged is_simulated=1
doctor_schedules     -- realistic weekly patterns (full-time/part-time mix)
branch_insurance     -- deterministic seed, not random noise
```

This isn't a shortcut hidden from the user — `context_builder.py` surfaces the distinction directly in responses: *"...I'd recommend confirming when you book, as coverage details can sometimes vary."* The chatbot is honest about what's confirmed fact versus reasonable estimate, which is the only acceptable behaviour for anything healthcare-adjacent.

---

## Observability — every query is fully traced

This isn't a black box. Every single message produces a `PipelineTrace` capturing intent confidence, which tables/chunks were hit, RRF scores, token counts, latency per stage, and cost in USD. Toggle `/debug` in the CLI or the web UI's debug panel to see it live:

```
① INTENT CLASSIFICATION
   Intent:       insurance_check  [Claude API, 312ms]
   Confidence:   ████████████████░░░░  82%
   entity.branch:    Festival City
   entity.insurance: NAS  (resolved from conversation history)

② SQL RETRIEVAL
   Tables hit:   insurance_providers, branch_insurance
   Rows:         1   |   Latency: 4ms

⑤ SUMMARY
   API calls: 2   |   Total: 1,528ms   |   Cost: $0.0069
```

Every turn also logs to `data/eval/eval_log.jsonl` — structured enough to compute aggregate accuracy, cost-per-intent, and latency percentiles without building a separate eval harness.

---

## ChromaDB Explorer

A standalone interactive tool for inspecting exactly what's in the vector store and why retrieval returns what it returns — built because "trust the embeddings" isn't a debugging strategy.

- **Embedding Space** — full corpus projected to 2D via PCA, D3.js scatter plot, colour-coded by source type, click any point for the underlying chunk + metadata
- **Browse Chunks** — paginated, filterable by source type / branch
- **Semantic Search** — type any query, see the actual chunks ChromaDB returns with cosine similarity scores, before any RRF fusion or LLM involvement

Run locally and open `/explorer`, or see it live in the [showcase page](https://bilalrazab.github.io/healthHubRag/SHOWCASE.html).

---

## Setup

```bash
git clone <repo-url> && cd healthhub-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY + FIRECRAWL_API_KEY
```

**Full pipeline (first run):**
```bash
python -m ingestion.scraper        # Firecrawl → data/raw/*.json
python -m ingestion.parser         # raw markdown → structured_data.json
python -m ingestion.db_loader      # structured_data.json → SQLite
python -m ingestion.vec_loader --reset   # → ChromaDB
```

**Run it:**
```bash
python -m interfaces.cli                          # terminal
uvicorn interfaces.api:app --reload --port 8000    # web UI at localhost:8000
```

**Deploy:** `render.yaml` blueprint included — connect repo on Render, select *Blueprint*, supply API keys. See [`DEPLOY.md`](./DEPLOY.md) for the full walkthrough including the path-resolution and memory fixes that came out of debugging a live deploy.

---

## Project structure

```
healthhub-rag/
├── ingestion/        scraper → parser → db_loader → vec_loader
├── rag/              intent · router · sql_retriever · vec_retriever
│                      · bm25_retriever · rrf · context_builder
├── chatbot/           bot (handle_message) · guards · prompt · session · debug
├── interfaces/        cli · api (FastAPI) · explore_api
├── static/            index.html (chat UI) · explore.html (ChromaDB viz)
├── data/
│   ├── structured/    structured_data.json — single source of truth
│   ├── db/            healthhub.db (SQLite)
│   ├── chroma/         vector index
│   └── eval/           eval_log.jsonl — every traced query
├── render.yaml         one-click Blueprint deploy
└── Dockerfile
```

---

## Stack

| Layer | Choice | Reasoning |
|---|---|---|
| Scraping | Firecrawl | Only option that handles fully client-rendered Elementor pages |
| Structured store | SQLite | Zero-ops for this scale; schema migrates cleanly to Postgres if needed |
| Vector store | ChromaDB | Local-first, metadata filtering built in, no external service dependency |
| Embeddings | `all-MiniLM-L6-v2` via ChromaDB's onnxruntime path | Same model as sentence-transformers, ~5x lighter at inference |
| Keyword search | `rank-bm25` | Classic BM25Okapi — exact-match recall vector search misses |
| Fusion | Reciprocal Rank Fusion (k=60) | Standard, score-free way to combine ranked lists from different retrievers |
| LLM | Claude Sonnet 4.6 | Both intent classification and final response generation |
| API | FastAPI | Async, typed, the same framework the eventual WhatsApp webhook will use |
| Deploy | Docker + Render | `render.yaml` blueprint = reproducible one-click deploy |

---

<div align="center">
