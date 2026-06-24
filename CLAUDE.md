# HealthHub RAG — Claude Code Instructions

## What This Project Is

A production-grade multi-branch healthcare RAG (Retrieval-Augmented Generation) system for **HealthHub by Al-Futtaim** — a 12-branch clinic network in Dubai. The system answers patient questions via terminal (V1) and WhatsApp (V2) using a hybrid structured SQL + semantic vector retrieval pipeline.

This is an **AI Engineer portfolio project** built by Bilal Razab. Every architectural decision must be explainable and defensible in a technical interview or video walkthrough.

---

## Tech Stack (locked — do not change)

| Layer | Tool |
|---|---|
| Language | Python 3.11.9 |
| Scraping | Firecrawl API (`firecrawl-py`) |
| Structured DB | SQLite (dev) → PostgreSQL (prod) |
| Vector DB | ChromaDB (local persistent) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (local, free) |
| Keyword search | `rank_bm25` (BM25Okapi) |
| Fusion | RRF — Reciprocal Rank Fusion |
| LLM | Claude `claude-sonnet-4-6` via Anthropic SDK |
| API server | FastAPI + uvicorn |
| WhatsApp | Meta Business API webhook |
| Env | `python-dotenv` |
| Testing | `pytest` |

**No LangChain. No LlamaIndex. No magic wrappers.** Every layer is written directly. The intelligence is in the design, not the abstraction.

---

## Project Structure

```
healthhub-rag/
│
├── CLAUDE.md                    ← YOU ARE HERE — read before touching anything
├── config.py                    ← Single source of truth for all settings
├── requirements.txt
├── .env                         ← Never commit. Contains all API keys.
├── .gitignore
│
├── ingestion/                   ← Everything about getting data IN
│   ├── scraper.py               ← Firecrawl-based scraper (JS-rendered pages)
│   ├── parser.py                ← Extracts structured entities from scraped markdown
│   ├── simulator.py             ← Generates fake schedules, insurance networks
│   ├── db_loader.py             ← Loads structured data into SQLite
│   └── vec_loader.py            ← Chunks text, embeds, upserts to ChromaDB
│
├── data/
│   ├── raw/                     ← Firecrawl output (markdown per page)
│   │   ├── branches/
│   │   ├── doctors/
│   │   ├── specialities/
│   │   ├── insurance/
│   │   ├── packages/
│   │   └── general/
│   ├── structured/              ← Parsed JSON ready for DB
│   ├── simulated/               ← Generated schedules + insurance
│   ├── db/                      ← healthhub.db (SQLite)
│   └── chroma/                  ← ChromaDB vector index
│
├── rag/
│   ├── intent.py                ← Intent classification + entity extraction (Claude)
│   ├── router.py                ← Routes to SQL | Vector | Hybrid | Rule
│   ├── sql_retriever.py         ← All structured SQL queries
│   ├── vec_retriever.py         ← ChromaDB semantic search + metadata filters
│   ├── bm25_retriever.py        ← BM25 keyword search
│   ├── rrf.py                   ← Reciprocal Rank Fusion
│   └── context_builder.py       ← Formats retrieval results for LLM
│
├── chatbot/
│   ├── bot.py                   ← handle_message() — THE core function
│   ├── prompt.py                ← System prompt + HealthHub persona
│   ├── session.py               ← Per-user conversation history
│   └── guards.py                ← Emergency detection, safety rules
│
├── adapters/
│   ├── base.py                  ← DataAdapter abstract interface
│   ├── sqlite_adapter.py        ← Current: reads from SQLite
│   └── api_adapter.py           ← Future: reads from HIS/CRM live API
│
├── interfaces/
│   ├── cli.py                   ← Terminal REPL (V1)
│   └── whatsapp.py              ← FastAPI webhook (V2)
│
├── tests/
│   ├── test_intents.py
│   ├── test_retrieval.py
│   └── test_conversations.py
│
└── scripts/
    ├── run_scraper.py
    ├── run_ingestion.py
    └── run_demo.py
```

---

## Environment Variables (.env)

```
ANTHROPIC_API_KEY=sk-ant-...
FIRECRAWL_API_KEY=fc-...
CLINIC_NAME=HealthHub by Al-Futtaim
CLINIC_URL=https://www.healthhubalfuttaim.com
ENVIRONMENT=development
WHATSAPP_TOKEN=
WHATSAPP_PHONE_ID=
WHATSAPP_VERIFY_TOKEN=healthhub_verify_2024
DB_PATH=data/db/healthhub.db
CHROMA_PATH=data/chroma
```

---

## Data Architecture

### Three-tier data model

| Tier | Source | Storage | Status |
|---|---|---|---|
| Scraped | Firecrawl → real website | SQLite + ChromaDB | Real data |
| Simulated | `simulator.py` script | SQLite | Placeholder for HIS API |
| Future hooks | Schema defined, empty | SQLite tables | API-ready |

### SQLite Schema (key tables)

```sql
branches        — id, name, area, address, phone, hours, maps_url, parking
doctors         — id, name, title, speciality_id, branch_id, languages, experience
specialities    — id, name, slug, description, conditions_treated
branch_specialities — branch_id, speciality_id
insurance_providers — id, name
branch_insurance    — branch_id, insurance_id (SIMULATED)
doctor_schedules    — doctor_id, branch_id, day_of_week, slot_start, slot_end (SIMULATED)
health_packages     — id, name, category, price_from, description
appointments        — (future — schema only)
```

### ChromaDB Metadata Schema

Every chunk stored in ChromaDB must have these metadata fields:

```python
{
  "source_type": "branch_page" | "speciality" | "blog" | "package" | "telehealth" | "faq" | "general",
  "branch":      "JVC" | "Al Karama" | ... | "all",
  "speciality":  "Cardiology" | "Physiotherapy" | ... | None,
  "language":    "en" | "ar",
  "page_type":   "clinical" | "admin" | "marketing" | "education",
  "has_price":   True | False,
  "doctor_name": "Dr. Aisha Rahman" | None,
  "chunk_id":    "unique-string"
}
```

---

## Intent Classification (12 classes)

| Intent | Route | Example |
|---|---|---|
| `branch_info` | SQL | "Where is Al Karama clinic?" |
| `branch_hours` | SQL | "Is Silicon Oasis open Sunday?" |
| `doctor_search` | SQL | "Find a cardiologist at JVC" |
| `doctor_availability` | SQL | "When is Dr. Aisha available?" |
| `speciality_info` | Vector (filtered) | "What does physiotherapy treat?" |
| `insurance_check` | SQL + Vector | "Do you take Daman at JVC?" |
| `appointment_guide` | Vector | "How do I book an appointment?" |
| `package_info` | SQL | "How much is the flu vaccine?" |
| `telehealth` | Vector | "Can I see a doctor online?" |
| `emergency` | RULE (hardcoded) | "I have chest pain" |
| `general_health` | Vector | "What causes high blood pressure?" |
| `complaint` | RULE (hardcoded) | "I want to give feedback" |

---

## Build Sequence (do NOT skip steps)

1. `ingestion/scraper.py` — Firecrawl scraper → `data/raw/`
2. `ingestion/parser.py` — Parse markdown → `data/structured/`
3. `ingestion/simulator.py` — Generate fake data → `data/simulated/`
4. `ingestion/db_loader.py` — Load all into SQLite
5. `ingestion/vec_loader.py` — Embed + index into ChromaDB
6. `rag/intent.py` + `rag/router.py` — Intent classification
7. `rag/sql_retriever.py` + `rag/vec_retriever.py` + `rag/rrf.py`
8. `chatbot/bot.py` + `chatbot/guards.py` + `chatbot/prompt.py`
9. `interfaces/cli.py` — Terminal demo
10. `interfaces/whatsapp.py` — WhatsApp webhook

---

## Coding Rules

- Always load `.env` via `python-dotenv` in `config.py`. Never hardcode keys.
- All file paths use `pathlib.Path`, never string concatenation.
- Every module has a single responsibility. No god files.
- Log with `logging`, not `print()`.
- All data classes use Python `dataclasses` with type annotations.
- SQLite connections opened with context managers (`with sqlite3.connect(...) as conn`).
- ChromaDB queries always include metadata `where` filters when a branch or speciality entity is extracted.
- The `handle_message()` function in `chatbot/bot.py` is transport-agnostic — CLI and WhatsApp both call it identically.
- Simulated data is clearly annotated in code with `# SIMULATED` comments.
- Future API hooks are annotated with `# FUTURE: replace with live API call`.
- Retry all external API calls (Firecrawl, Anthropic, WhatsApp) with exponential backoff.

---

## Branch Reference (all 12)

| Slug | Display Name | Area |
|---|---|---|
| healthhub-al-karama | HealthHub – Al Karama | Al Karama |
| healthhub-al-nahda | HealthHub – Al Nahda | Al Nahda |
| healthhub-al-qusais | HealthHub – Al Qusais | Al Qusais |
| healthhub-al-warqa | HealthHub – Al Warqa | Al Warqa |
| healthhub-arabian-center | HealthHub – Arabian Center | Arabian Center |
| healthhub-barsha-heights | HealthHub – Barsha Heights | Barsha Heights |
| healthhub-festival-plaza | HealthHub – Festival Plaza | Festival City |
| international-city | HealthHub – International City | International City |
| silicon-oasis | HealthHub – Silicon Oasis | Silicon Oasis |
| discovery-gardens | HealthHub – Discovery Gardens | Discovery Gardens |
| healthhub-jvc-jumeirah-village-circle | HealthHub – JVC | JVC |
| day-surgery-center | HealthHub Day Surgery | Festival City |

---

## Known Insurance Providers (20)

AbuDhabi National Insurance, Al Buhaira National Insurance, Almadallah, Aspire, Daman, Ecare, FMC, GIG Insurance, GlobeMed, Inayah, Mednet, Metlife, MSH International, NAS, National General Insurance, Neuron, NextCare, Now Health, Saudi Arabian Insurance, Sukoon Insurance

---

## Portfolio Context

This project demonstrates:
- System design (multi-layer, adapter pattern, future API hooks)
- Data engineering (Firecrawl, parsing, SQLite, ChromaDB)
- NLP pipeline (intent classification, entity extraction, hybrid retrieval, RRF)
- LLM engineering (prompt design, grounded responses, safety guards)
- Backend (FastAPI, async, webhook)
- Production thinking (simulated data explained, future integration hooks)

**Video series companion:** Each build step maps to one episode. The architecture decisions and tradeoffs must be clearly explainable on camera.
