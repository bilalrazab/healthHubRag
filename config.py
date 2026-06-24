"""
config.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single source of truth for all project settings.
Import from here everywhere — never duplicate constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

# ── Paths ─────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
RAW_DIR    = DATA_DIR / "raw"
STRUCT_DIR = DATA_DIR / "structured"
SIM_DIR    = DATA_DIR / "simulated"
DB_PATH    = Path(os.getenv("DB_PATH", str(DATA_DIR / "db" / "healthhub.db")))
CHROMA_DIR = Path(os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma")))

# ── API Keys ──────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")

if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set.\n"
        "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
    )

if not FIRECRAWL_API_KEY:
    raise EnvironmentError(
        "FIRECRAWL_API_KEY not set.\n"
        "Get one at https://firecrawl.dev and add to .env: FIRECRAWL_API_KEY=fc-..."
    )

# ── Clinic ────────────────────────────────────────────────────
CLINIC_NAME = os.getenv("CLINIC_NAME", "HealthHub by Al-Futtaim")
CLINIC_URL  = os.getenv("CLINIC_URL", "https://www.healthhubalfuttaim.com")

# ── Scraping ──────────────────────────────────────────────────
SCRAPE_DELAY = 1.0  # seconds between Firecrawl requests

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 80    # overlap between chunks

# ── Embeddings ────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── ChromaDB ──────────────────────────────────────────────────
CHROMA_COLLECTION = "healthhub_docs"

# ── Retrieval ─────────────────────────────────────────────────
TOP_K_SEMANTIC = 8
TOP_K_KEYWORD  = 8
TOP_K_FINAL    = 5
RRF_K          = 60

# ── LLM ───────────────────────────────────────────────────────
CLAUDE_MODEL  = "claude-sonnet-4-6"
MAX_TOKENS    = 600
TEMPERATURE   = 0.3

# ── WhatsApp ──────────────────────────────────────────────────
WHATSAPP_TOKEN        = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_ID     = os.getenv("WHATSAPP_PHONE_ID", "")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "healthhub_verify_2024")

# ── Branch slugs (all 12) ─────────────────────────────────────
BRANCH_SLUGS = [
    "healthhub-al-karama",
    "healthhub-al-nahda",
    "healthhub-al-qusais",
    "healthhub-al-warqa",
    "healthhub-arabian-center",
    "healthhub-barsha-heights",
    "healthhub-festival-plaza",
    "international-city",
    "silicon-oasis",
    "discovery-gardens",
    "healthhub-jvc-jumeirah-village-circle",
    "day-surgery-center",
]

# ── Speciality slugs (all 25) ─────────────────────────────────
SPECIALITY_SLUGS = [
    "anaesthesiology",
    "cardiology",
    "clinical-dietitian-and-nutritionist",
    "dentistry",
    "dermatology",
    "endocrinology",
    "ent",
    "family-medicine",
    "gastroenterology",
    "general-medicine",
    "general-surgery",
    "hair-transplant",
    "internal-medicine",
    "neurology",
    "neurosurgery",
    "obstetrics-gynaecology",
    "ophthalmology",
    "orthopaedics",
    "paediatrics",
    "pathology-medical-speciality-services-in-dubai-healthhub-by-al-futtaim",
    "physiotherapy",
    "psychiatry",
    "radiology",
    "urology",
    "vascular-surgery",
]

# ── General pages ─────────────────────────────────────────────
GENERAL_PAGES = [
    ("about",          "/about-us/"),
    ("telehealth",     "/telehealth/"),
    ("appointment",    "/new-appointment/"),
    ("insurance",      "/insurance-provider/"),
    ("packages",       "/health-packages/"),
    ("patient_rights", "/patient-rights-and-responsibilities/"),
    ("privacy",        "/privacy-policy-rights/"),
]

# ── Insurance providers (confirmed list) ──────────────────────
INSURANCE_PROVIDERS = [
    "AbuDhabi National Insurance Company",
    "Al Buhaira National Insurance Company",
    "Almadallah",
    "Aspire",
    "Daman Insurance Company",
    "Ecare",
    "FMC",
    "GIG Insurance Company",
    "GlobeMed",
    "Inayah",
    "Mednet",
    "Metlife",
    "MSH International",
    "NAS",
    "National General Insurance",
    "Neuron",
    "NextCare",
    "Now Health",
    "Saudi Arabian Insurance Company",
    "Sukoon Insurance",
]
