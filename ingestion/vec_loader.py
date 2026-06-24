"""
ingestion/vec_loader.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HealthHub — ChromaDB Vector Loader

Reads: data/structured/structured_data.json
Indexes ALL entity types into ChromaDB with rich metadata.

What this fixes vs original vector_loader.py:
  1. Indexes branches, doctors, packages (not just text_content)
  2. Every chunk has branch, speciality, doctor_name metadata
     → enables pre-filtered semantic search
  3. Paragraph-aware chunking (word-based, not character-based)
  4. Uses sentence-transformers locally (not ChromaDB default embedder)
  5. Cleans dirty data before indexing (junk in nationality, etc.)

Run:
    python -m ingestion.vec_loader
    python -m ingestion.vec_loader --reset   (wipe and rebuild)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    STRUCT_DIR, CHROMA_DIR, CHROMA_COLLECTION,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vec_loader")

STRUCTURED_FILE = STRUCT_DIR / "structured_data.json"
EMBED_BATCH  = 64
CHROMA_BATCH = 500

# ── Chunking ─────────────────────────────────────────────────

def chunk_paragraphs(text: str,
                     chunk_size: int = CHUNK_SIZE,
                     overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Paragraph-aware chunking at word level.
    Preserves sentence/fact boundaries.
    chunk_size = max words per chunk (not characters).
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()
        if current_words and len(current_words) + len(para_words) > chunk_size:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap:] if overlap else []
        current_words.extend(para_words)
        while len(current_words) > chunk_size:
            chunks.append(" ".join(current_words[:chunk_size]))
            current_words = current_words[chunk_size - overlap:]

    if current_words:
        final = " ".join(current_words).strip()
        if final:
            chunks.append(final)

    # Drop micro-chunks (< 15 words — not meaningful for retrieval)
    return [c for c in chunks if len(c.split()) >= 15]


def _cid(url: str, suffix: str, idx: int) -> str:
    """Stable deterministic chunk ID."""
    key = f"{url}_{suffix}_{idx}"
    return "doc_" + hashlib.md5(key.encode()).hexdigest() + f"_{idx:04d}"


# ── Data cleaning ─────────────────────────────────────────────

def _clean_nationality(raw: str) -> str:
    """"Jordan\n\nNationality\n\nClinics" → "Jordan" """
    if not raw:
        return ""
    return raw.strip().split("\n")[0].strip()


def _short_branch(name: str) -> str:
    """
    "HealthHub – Al Karama" → "Al Karama"
    Used as metadata filter value for compact matching.
    """
    return re.sub(r"^HealthHub[\s\–\-–—]+", "", name).strip()


def _is_junk(text: str) -> bool:
    """Filter out image paths and non-content strings."""
    if not text or len(text) < 5:
        return True
    if text.startswith("content/") or text.startswith("http"):
        return True
    if text.endswith(")") and "uploads" in text:
        return True
    return False


# ── Text builders ─────────────────────────────────────────────

def _branch_to_text(b: dict) -> str:
    parts = []
    name = b.get("name", "")
    if name:
        parts.append(f"Clinic: {name}")
    overview = b.get("overview", b.get("description", ""))
    if overview:
        parts.append(overview)
    specs = b.get("specialities", [])
    if specs:
        parts.append(f"Specialities at this branch: {', '.join(specs)}")
    # Hours are the same for all branches
    parts.append("Opening hours: 8:00 AM to 10:00 PM, open every day of the week.")
    return "\n\n".join(parts)


def _doctor_to_text(d: dict) -> str:
    parts = []
    name = d.get("name", "")
    if name:
        parts.append(f"Doctor: {name}")
    title = d.get("title", "")
    if title:
        parts.append(f"Title: {title}")
    exp = d.get("experience_years")
    if exp:
        parts.append(f"Experience: {exp} years")
    nationality = _clean_nationality(d.get("nationality", ""))
    if nationality:
        parts.append(f"Nationality: {nationality}")
    langs = d.get("languages", [])
    if langs:
        parts.append(f"Languages spoken: {', '.join(langs)}")
    about = d.get("about", "")
    if about:
        parts.append(f"About: {about}")
    expertise = d.get("expertise", [])
    if expertise:
        parts.append(f"Areas of expertise: {', '.join(expertise)}")
    clinics = d.get("clinics", [])
    if clinics:
        parts.append(f"Available at: {', '.join(clinics)}")
    return "\n\n".join(parts)


def _speciality_to_text(s: dict) -> str:
    """
    text_content items with type='speciality'
    Already have a clean_body from the parser.
    """
    title = s.get("title", "")
    body  = s.get("clean_body", "")
    # Strip image markdown
    body  = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", body).strip()
    return f"Speciality: {title}\n\n{body}" if title else body


def _package_to_text(p: dict) -> str:
    parts = []
    name = p.get("package_name", "")
    if name and name != "Health Packages":
        parts.append(f"Health package: {name}")
    price = str(p.get("price", "") or "")
    if price:
        parts.append(f"Price: AED {price}")
    category = p.get("category", "")
    if category:
        parts.append(f"Category: {category}")
    inclusions = [
        i for i in p.get("inclusions", [])
        if not _is_junk(i)
    ]
    if inclusions:
        parts.append("Includes:\n" + "\n".join(f"- {i}" for i in inclusions[:15]))
    return "\n\n".join(parts)


def _general_to_text(g: dict) -> str:
    title = g.get("title", "")
    body  = g.get("clean_body", "")
    body  = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", body).strip()
    return f"# {title}\n\n{body}" if title else body


# ── Metadata builders ─────────────────────────────────────────

def _branch_meta(b: dict, idx: int) -> dict:
    short = _short_branch(b.get("name", ""))
    return {
        "source_type": "branch_page",
        "branch":      short,
        "speciality":  "all",
        "doctor_name": "none",
        "language":    "en",
        "page_type":   "clinical",
        "has_price":   "false",
        "chunk_index": str(idx),
        "url":         b.get("url", ""),
        "title":       b.get("name", ""),
    }


def _doctor_meta(d: dict, idx: int) -> dict:
    clinics = d.get("clinics", [])
    branch  = _short_branch(clinics[0]) if clinics else "all"
    # Infer speciality from title
    title   = d.get("title", "")
    spec    = _infer_speciality(title)
    return {
        "source_type": "doctor_profile",
        "branch":      branch,
        "speciality":  spec or "none",
        "doctor_name": d.get("name", ""),
        "language":    "en",
        "page_type":   "clinical",
        "has_price":   "false",
        "chunk_index": str(idx),
        "url":         d.get("url", ""),
        "title":       d.get("name", ""),
    }


def _speciality_meta(s: dict, idx: int) -> dict:
    return {
        "source_type": "speciality",
        "branch":      "all",
        "speciality":  s.get("title", ""),
        "doctor_name": "none",
        "language":    "en",
        "page_type":   "clinical",
        "has_price":   "false",
        "chunk_index": str(idx),
        "url":         s.get("url", ""),
        "title":       s.get("title", ""),
    }


def _package_meta(p: dict, idx: int) -> dict:
    price = str(p.get("price", "") or "")
    return {
        "source_type": "package",
        "branch":      "all",
        "speciality":  "none",
        "doctor_name": "none",
        "language":    "en",
        "page_type":   "admin",
        "has_price":   "true" if price else "false",
        "chunk_index": str(idx),
        "url":         p.get("url", ""),
        "title":       p.get("package_name", ""),
    }


def _general_meta(g: dict, idx: int) -> dict:
    return {
        "source_type": g.get("type", "general"),
        "branch":      "all",
        "speciality":  "none",
        "doctor_name": "none",
        "language":    "en",
        "page_type":   "general",
        "has_price":   "false",
        "chunk_index": str(idx),
        "url":         g.get("url", ""),
        "title":       g.get("title", ""),
    }


# ── Speciality inference ──────────────────────────────────────

_SPEC_KEYWORDS = {
    "Cardiology":               ["cardiolog"],
    "Dermatology":              ["dermatolog", "cosmetolog"],
    "Paediatrics":              ["paediatric", "pediatric"],
    "Orthopaedics":             ["orthopaed", "orthoped"],
    "Obstetrics & Gynaecology": ["obstetric", "gynaecol", "gynecol"],
    "Internal Medicine":        ["internal medicine"],
    "Family Medicine":          ["family medicine"],
    "General Medicine":         ["general medicine"],
    "General Practice":         ["general practice"],
    "General Surgery":          ["general surgeon", "general surgery"],
    "Physiotherapy":            ["physiotherap"],
    "ENT":                      ["ent", "otolaryngol"],
    "Urology":                  ["urolog"],
    "Gastroenterology":         ["gastroenterolog"],
    "Endocrinology":            ["endocrinolog"],
    "Radiology":                ["radiolog"],
    "Anaesthesiology":          ["anaesthesiolog", "anesthesiolog"],
    "Neurology":                ["neurolog"],
    "Neurosurgery":             ["neurosurger"],
    "Ophthalmology":            ["ophthalmolog"],
    "Vascular Surgery":         ["vascular"],
    "Psychiatry":               ["psychiatr"],
    "Hair Transplant":          ["hair transplant"],
    "Pathology":                ["patholog"],
    "Dentistry":                ["dentist", "dental", "implantolog"],
    "Dietetics":                ["dietitian", "nutritionist"],
}


def _infer_speciality(title: str) -> str:
    t = title.lower()
    for spec, keywords in _SPEC_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return spec
    return ""


# ── Main ─────────────────────────────────────────────────────

def run(reset: bool = False) -> None:
    log.info("═" * 56)
    log.info("  HealthHub Vector Loader")
    log.info("  Model:      %s", EMBEDDING_MODEL)
    log.info("  Collection: %s", CHROMA_COLLECTION)
    log.info("  Input:      %s", STRUCTURED_FILE)
    log.info("═" * 56)

    if not STRUCTURED_FILE.exists():
        raise FileNotFoundError(
            f"structured_data.json not found at {STRUCTURED_FILE}"
        )

    data = json.loads(STRUCTURED_FILE.read_text(encoding="utf-8"))
    log.info("  Loaded: %d doctors, %d branches, %d packages, %d text_content",
             len(data.get("doctors", [])),
             len(data.get("branches", [])),
             len(data.get("packages", [])),
             len(data.get("text_content", [])))

    # Load model
    log.info("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info("  Model ready.")

    # Connect ChromaDB
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
            log.info("  Deleted existing collection.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not reset:
        log.warning("Collection has %d chunks. Use --reset to rebuild.", collection.count())
        return

    all_texts:  list[str]  = []
    all_metas:  list[dict] = []
    all_ids:    list[str]  = []

    # ── 1. Branches ───────────────────────────────────────────
    for b in data.get("branches", []):
        text = _branch_to_text(b)
        url  = b.get("url", b.get("name", ""))
        for i, chunk in enumerate(chunk_paragraphs(text)):
            all_texts.append(chunk)
            all_metas.append(_branch_meta(b, i))
            all_ids.append(_cid(url, "branch", i))
    log.info("  Branch chunks: %d", len(all_texts))
    n_branches = len(all_texts)

    # ── 2. Doctors ────────────────────────────────────────────
    for d in data.get("doctors", []):
        text = _doctor_to_text(d)
        url  = d.get("url", d.get("name", ""))
        for i, chunk in enumerate(chunk_paragraphs(text)):
            all_texts.append(chunk)
            all_metas.append(_doctor_meta(d, i))
            all_ids.append(_cid(url, "doctor", i))
    log.info("  Doctor chunks: %d", len(all_texts) - n_branches)
    n_doctors = len(all_texts)

    # ── 3. Specialities ───────────────────────────────────────
    specialities = [
        t for t in data.get("text_content", [])
        if t.get("type") == "speciality"
    ]
    for s in specialities:
        text = _speciality_to_text(s)
        url  = s.get("url", s.get("title", ""))
        for i, chunk in enumerate(chunk_paragraphs(text)):
            all_texts.append(chunk)
            all_metas.append(_speciality_meta(s, i))
            all_ids.append(_cid(url, "spec", i))
    log.info("  Speciality chunks: %d", len(all_texts) - n_doctors)
    n_specs = len(all_texts)

    # ── 4. Packages ───────────────────────────────────────────
    packages = [
        p for p in data.get("packages", [])
        if p.get("package_name", "") not in ("Health Packages",)
    ]
    for p in packages:
        text = _package_to_text(p)
        url  = p.get("url", p.get("package_name", ""))
        for i, chunk in enumerate(chunk_paragraphs(text)):
            all_texts.append(chunk)
            all_metas.append(_package_meta(p, i))
            all_ids.append(_cid(url, "pkg", i))
    log.info("  Package chunks: %d", len(all_texts) - n_specs)
    n_pkgs = len(all_texts)

    # ── 5. General pages ──────────────────────────────────────
    general = [
        t for t in data.get("text_content", [])
        if t.get("type") != "speciality"
    ]
    for g in general:
        text = _general_to_text(g)
        url  = g.get("url", g.get("key", ""))
        for i, chunk in enumerate(chunk_paragraphs(text)):
            all_texts.append(chunk)
            all_metas.append(_general_meta(g, i))
            all_ids.append(_cid(url, "gen", i))
    log.info("  General chunks: %d", len(all_texts) - n_pkgs)

    total = len(all_texts)
    log.info("  Total chunks: %d", total)

    if total == 0:
        log.error("No chunks produced. Check structured_data.json.")
        return

    # Deduplicate IDs
    seen = set()
    texts, metas, ids = [], [], []
    for t, m, i in zip(all_texts, all_metas, all_ids):
        if i not in seen:
            seen.add(i)
            texts.append(t)
            metas.append(m)
            ids.append(i)
    log.info("  After dedup: %d chunks", len(ids))

    # Embed
    log.info("Embedding chunks...")
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        embs  = model.encode(batch, normalize_embeddings=True,
                             show_progress_bar=False)
        embeddings.extend(embs.tolist())
        log.info("  Embedded %d / %d", min(i + EMBED_BATCH, len(texts)), len(texts))

    # Upsert
    log.info("Upserting to ChromaDB...")
    for i in range(0, len(ids), CHROMA_BATCH):
        collection.upsert(
            ids=ids[i:i + CHROMA_BATCH],
            documents=texts[i:i + CHROMA_BATCH],
            embeddings=embeddings[i:i + CHROMA_BATCH],
            metadatas=metas[i:i + CHROMA_BATCH],
        )
        log.info("  Upserted %d / %d", min(i + CHROMA_BATCH, len(ids)), len(ids))

    log.info("═" * 56)
    log.info("  ChromaDB: %d chunks indexed", collection.count())
    log.info("  Location: %s", CHROMA_DIR)
    log.info("═" * 56)
    log.info("🎉 Vector store ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HealthHub Vector Loader")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe and rebuild ChromaDB collection")
    args = parser.parse_args()
    run(reset=args.reset)