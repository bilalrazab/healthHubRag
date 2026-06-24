"""
ingestion/db_loader.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HealthHub — SQLite Loader

Reads: data/structured/structured_data.json (single combined file)
Writes: data/db/healthhub.db

Handles all data quality issues found in the real data:
  - Nationality field contains junk: "Jordan\n\nNationality\n\nClinics"
  - Branch names use en-dash (–), doctor clinic names use hyphen (-)
  - Insurance has junk entries like "All Insurance Providers", "provider/)"
  - Packages has duplicate index-page entries
  - Doctor → Branch linking needs fuzzy normalised matching

Run:
    python -m ingestion.db_loader
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import logging
import random
import re
import sqlite3
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH, STRUCT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("db_loader")

STRUCTURED_FILE = STRUCT_DIR / "structured_data.json"

DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday"]

# Realistic schedule patterns: (start, end, days)
SLOT_PATTERNS = [
    ("08:00", "14:00", ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]),
    ("14:00", "22:00", ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]),
    ("08:00", "22:00", ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]),
    ("10:00", "16:00", ["Monday", "Wednesday", "Friday"]),
    ("09:00", "13:00", ["Tuesday", "Thursday"]),
    ("08:00", "14:00", ["Saturday"]),
    ("08:00", "22:00", ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Saturday"]),
]

# ── Schema ────────────────────────────────────────────────────

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS branches (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT    NOT NULL,
    url                 TEXT,
    description         TEXT,
    address             TEXT,
    phone               TEXT,
    hours               TEXT    DEFAULT '8:00 AM – 10:00 PM daily',
    parking             TEXT,
    maps_url            TEXT,
    specialities_json   TEXT
);

CREATE TABLE IF NOT EXISTS specialities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    UNIQUE NOT NULL,
    slug        TEXT,
    description TEXT,
    tagline     TEXT
);

CREATE TABLE IF NOT EXISTS branch_specialities (
    branch_id     INTEGER REFERENCES branches(id)     ON DELETE CASCADE,
    speciality_id INTEGER REFERENCES specialities(id) ON DELETE CASCADE,
    PRIMARY KEY (branch_id, speciality_id)
);

CREATE TABLE IF NOT EXISTS doctors (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT    NOT NULL,
    url              TEXT,
    title            TEXT,
    speciality       TEXT,
    experience_years INTEGER,
    nationality      TEXT,
    languages_json   TEXT,
    expertise_json   TEXT,
    about            TEXT
);

CREATE TABLE IF NOT EXISTS doctor_branches (
    doctor_id  INTEGER REFERENCES doctors(id)  ON DELETE CASCADE,
    branch_id  INTEGER REFERENCES branches(id) ON DELETE CASCADE,
    PRIMARY KEY (doctor_id, branch_id)
);

-- SIMULATED: replace with live HIS API data in production
CREATE TABLE IF NOT EXISTS doctor_schedules (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id    INTEGER REFERENCES doctors(id)  ON DELETE CASCADE,
    branch_id    INTEGER REFERENCES branches(id) ON DELETE CASCADE,
    day_of_week  TEXT    NOT NULL,
    slot_start   TEXT    NOT NULL,
    slot_end     TEXT    NOT NULL,
    is_simulated INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS insurance_providers (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

-- SIMULATED: branch-level insurance varies in reality
CREATE TABLE IF NOT EXISTS branch_insurance (
    branch_id    INTEGER REFERENCES branches(id)            ON DELETE CASCADE,
    insurance_id INTEGER REFERENCES insurance_providers(id) ON DELETE CASCADE,
    is_simulated INTEGER DEFAULT 1,
    PRIMARY KEY (branch_id, insurance_id)
);

CREATE TABLE IF NOT EXISTS health_packages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    url         TEXT,
    price_text  TEXT,
    category    TEXT,
    description TEXT
);
"""

# ── Helpers ───────────────────────────────────────────────────

def _load_data() -> dict:
    if not STRUCTURED_FILE.exists():
        raise FileNotFoundError(
            f"structured_data.json not found at {STRUCTURED_FILE}\n"
            "Run the parser first: python -m ingestion.parser"
        )
    return json.loads(STRUCTURED_FILE.read_text(encoding="utf-8"))


def _clean_nationality(raw: str) -> str:
    """
    Raw nationality field: "Jordan\n\nNationality\n\nClinics"
    Extract just the country name (first non-empty line).
    """
    if not raw:
        return ""
    first = raw.strip().split("\n")[0].strip()
    # Skip if it's a markdown label
    if first.lower() in ("nationality", "clinics", "languages", ""):
        return ""
    return first


def _normalise_branch_name(name: str) -> str:
    """
    Normalise branch name for matching.
    Converts en-dash (–) to hyphen (-), lowercases, strips extra spaces.
    Branch names: "HealthHub – Al Karama"
    Doctor clinic names: "HealthHub - Al Karama"
    Both → "healthhub - al karama"
    """
    return name.replace("–", "-").replace("—", "-").strip().lower()


def _build_branch_map(conn: sqlite3.Connection) -> dict:
    """
    Returns a dict mapping normalised branch name → branch_id.
    Includes extra variants (e.g. short names, JVC alias).
    """
    rows = conn.execute("SELECT id, name FROM branches").fetchall()
    m = {}
    for row in rows:
        bid  = row[0]
        name = row[1]
        # Primary key: normalised full name
        m[_normalise_branch_name(name)] = bid
        # Also map the part after "HealthHub – " or "HealthHub - "
        short = re.sub(r"^healthhub[\s\-–]+", "", _normalise_branch_name(name)).strip()
        m[short] = bid
    # Manual aliases for common mismatches found in data
    aliases = {
        "healthhub day surgery - festival city":      "healthhub day surgery – festival city",
        "healthhub - jvc (jumeirah village circle)":  "healthhub – jvc (jumeirah village circle)",
        "healthhub - jumeirah village circle":        "healthhub – jvc (jumeirah village circle)",
        "healthhub - arabian center":                 "healthhub – arabian center",
    }
    for alias, canonical in aliases.items():
        if canonical in m:
            m[alias] = m[canonical]
    return m


def _resolve_branch(clinic_name: str, branch_map: dict) -> int | None:
    """
    Match a doctor's clinic name string to a branch_id.
    Tries normalised exact match first, then substring fuzzy match.
    """
    norm = _normalise_branch_name(clinic_name)
    if norm in branch_map:
        return branch_map[norm]
    # Fuzzy: check if any branch key is contained in the clinic name or vice versa
    for key, bid in branch_map.items():
        if key and (key in norm or norm in key):
            return bid
    return None


def _is_junk_insurance(name: str) -> bool:
    """Filter out non-insurance-provider strings from the scraped insurance list."""
    if not name or len(name) < 4:
        return True
    junk = {
        "all insurance providers", "insurance partners",
        "fmc",  # too ambiguous — keep if needed
    }
    if name.lower() in junk:
        return True
    # Strings that look like partial URLs or image paths
    if name.startswith("content/") or name.endswith(")") or "http" in name:
        return True
    return False


# ── Table loaders ─────────────────────────────────────────────

def load_branches(conn: sqlite3.Connection, data: dict) -> None:
    branches = data.get("branches", [])
    log.info("  Loading %d branches...", len(branches))
    for b in branches:
        name        = b.get("name", "")
        url         = b.get("url", "")
        description = b.get("overview", "")
        specs       = b.get("specialities", [])
        specs_json  = json.dumps(specs)

        conn.execute("""
            INSERT INTO branches (name, url, description, hours, specialities_json)
            VALUES (?, ?, ?, ?, ?)
        """, (name, url, description, "8:00 AM – 10:00 PM daily", specs_json))

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM branches").fetchone()[0]
    log.info("  ✓ Branches: %d rows", count)


def load_specialities(conn: sqlite3.Connection, data: dict) -> None:
    """
    Extract specialities from two sources:
    1. text_content items with type='speciality' (full descriptions)
    2. branch specialities lists (names only — fill any gaps)
    """
    # Source 1: full speciality descriptions
    spec_descriptions = {
        item["title"]: item.get("clean_body", "")
        for item in data.get("text_content", [])
        if item.get("type") == "speciality"
    }

    # Source 2: all unique names from branch lists
    all_spec_names = set()
    for b in data.get("branches", []):
        for s in b.get("specialities", []):
            if s:
                all_spec_names.add(s)

    # Merge: use description from text_content if available
    inserted = 0
    for name in sorted(all_spec_names):
        slug = name.lower().replace(" ", "-").replace("&", "and").replace(",", "")
        desc = spec_descriptions.get(name, "")
        # Clean the description (remove markdown image tags)
        desc = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", desc).strip()
        try:
            conn.execute("""
                INSERT OR IGNORE INTO specialities (name, slug, description)
                VALUES (?, ?, ?)
            """, (name, slug, desc))
            inserted += 1
        except Exception as e:
            log.warning("Speciality insert error (%s): %s", name, e)

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM specialities").fetchone()[0]
    log.info("  ✓ Specialities: %d rows", count)


def load_branch_specialities(conn: sqlite3.Connection, data: dict) -> None:
    """Link each branch to its specialities."""
    branch_map = _build_branch_map(conn)
    # spec name → spec_id
    spec_rows = conn.execute("SELECT id, name FROM specialities").fetchall()
    spec_map = {row[1].lower(): row[0] for row in spec_rows}

    count = 0
    for b in data.get("branches", []):
        branch_id = _resolve_branch(b["name"], branch_map)
        if not branch_id:
            log.warning("  Branch not found in DB: %s", b["name"])
            continue
        for spec_name in b.get("specialities", []):
            spec_id = spec_map.get(spec_name.lower())
            if spec_id:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO branch_specialities (branch_id, speciality_id)
                        VALUES (?, ?)
                    """, (branch_id, spec_id))
                    count += 1
                except Exception:
                    pass

    conn.commit()
    log.info("  ✓ Branch↔Speciality links: %d", count)


def load_doctors(conn: sqlite3.Connection, data: dict) -> None:
    branch_map = _build_branch_map(conn)
    doctors = data.get("doctors", [])
    log.info("  Loading %d doctors...", len(doctors))

    loaded = 0
    for d in doctors:
        name       = d.get("name", "")
        url        = d.get("url", "")
        title      = d.get("title", "")
        exp        = d.get("experience_years")
        # Infer speciality from title
        speciality = _infer_speciality(title)
        # Clean nationality — real data has junk appended
        nationality = _clean_nationality(d.get("nationality", ""))
        langs      = json.dumps(d.get("languages", []))
        expertise  = json.dumps(d.get("expertise", []))
        about      = d.get("about", "")

        try:
            cursor = conn.execute("""
                INSERT INTO doctors
                    (name, url, title, speciality, experience_years,
                     nationality, languages_json, expertise_json, about)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, url, title, speciality, exp,
                  nationality, langs, expertise, about))
            doctor_id = cursor.lastrowid
            loaded += 1
        except Exception as e:
            log.warning("Doctor insert error (%s): %s", name, e)
            continue

        # Link doctor → branches
        for clinic_name in d.get("clinics", []):
            branch_id = _resolve_branch(clinic_name, branch_map)
            if branch_id:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO doctor_branches (doctor_id, branch_id)
                        VALUES (?, ?)
                    """, (doctor_id, branch_id))
                except Exception:
                    pass

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM doctors").fetchone()[0]
    links = conn.execute("SELECT COUNT(*) FROM doctor_branches").fetchone()[0]
    log.info("  ✓ Doctors: %d rows, %d branch links", count, links)


def simulate_schedules(conn: sqlite3.Connection) -> None:
    """
    Generate realistic but fake weekly schedules for every doctor×branch pair.
    SIMULATED — flagged with is_simulated=1.
    Replace with live HIS API data in production.
    """
    pairs = conn.execute(
        "SELECT doctor_id, branch_id FROM doctor_branches"
    ).fetchall()

    random.seed(42)  # reproducible
    count = 0
    for doctor_id, branch_id in pairs:
        start, end, days = random.choice(SLOT_PATTERNS)
        for day in days:
            conn.execute("""
                INSERT INTO doctor_schedules
                    (doctor_id, branch_id, day_of_week, slot_start, slot_end, is_simulated)
                VALUES (?, ?, ?, ?, ?, 1)
            """, (doctor_id, branch_id, day, start, end))
            count += 1

    conn.commit()
    log.info("  ✓ Doctor schedules: %d slots (SIMULATED)", count)


def load_insurance(conn: sqlite3.Connection, data: dict) -> dict:
    """
    Extract insurance provider names from scraped data + confirmed fallback list.
    Returns {name_lower: id} map.
    """
    # Extract from scraped data, filtering junk
    scraped_names = []
    for i in data.get("insurance", []):
        for name in i.get("accepted_networks", []):
            if not _is_junk_insurance(name):
                scraped_names.append(name)

    # Confirmed fallback list (from config)
    from config import INSURANCE_PROVIDERS
    confirmed = INSURANCE_PROVIDERS

    # Merge: scraped first, then fill from confirmed if missing
    all_names = list(dict.fromkeys(scraped_names))  # dedup, preserve order
    for name in confirmed:
        if name not in all_names:
            all_names.append(name)

    insurance_map = {}
    for name in all_names:
        try:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO insurance_providers (name) VALUES (?)", (name,)
            )
            if cursor.lastrowid:
                insurance_map[name.lower()] = cursor.lastrowid
        except Exception:
            pass

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM insurance_providers").fetchone()[0]
    log.info("  ✓ Insurance providers: %d", count)
    return insurance_map


def simulate_branch_insurance(conn: sqlite3.Connection) -> None:
    """
    Assign insurers to branches.
    Every branch gets Daman, NAS, NextCare.
    70% chance of each other insurer per branch.
    SIMULATED — flagged with is_simulated=1.
    """
    branches  = conn.execute("SELECT id FROM branches").fetchall()
    insurers  = conn.execute("SELECT id, name FROM insurance_providers").fetchall()

    guaranteed = {"daman insurance company", "nas", "nextcare"}
    random.seed(42)
    count = 0

    for (branch_id,) in branches:
        for ins_id, ins_name in insurers:
            if ins_name.lower() in guaranteed or random.random() < 0.7:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO branch_insurance
                            (branch_id, insurance_id, is_simulated)
                        VALUES (?, ?, 1)
                    """, (branch_id, ins_id))
                    count += 1
                except Exception:
                    pass

    conn.commit()
    log.info("  ✓ Branch↔Insurance links: %d (SIMULATED)", count)


def load_packages(conn: sqlite3.Connection, data: dict) -> None:
    packages = data.get("packages", [])
    # Skip index page entries (no real slug, duplicate names)
    packages = [
        p for p in packages
        if p.get("package_name", "") not in ("Health Packages",)
        and p.get("url", "").rstrip("/") != "https://www.healthhubalfuttaim.com/health-packages"
    ]
    # Also deduplicate by name
    seen_names = set()
    unique = []
    for p in packages:
        name = p.get("package_name", "")
        if name and name not in seen_names:
            seen_names.add(name)
            unique.append(p)

    for p in unique:
        name     = p.get("package_name", "")
        url      = p.get("url", "")
        price    = str(p.get("price", "") or "")
        category = p.get("category", "")
        desc     = " ".join([
            i for i in p.get("inclusions", [])
            if not i.startswith("content/") and len(i) > 5
        ])
        try:
            conn.execute("""
                INSERT OR IGNORE INTO health_packages
                    (name, url, price_text, category, description)
                VALUES (?, ?, ?, ?, ?)
            """, (name, url, price, category, desc))
        except Exception as e:
            log.warning("Package insert error (%s): %s", name, e)

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM health_packages").fetchone()[0]
    log.info("  ✓ Health packages: %d rows", count)


# ── Speciality inference ──────────────────────────────────────

_SPEC_KEYWORDS = {
    "Cardiology":               ["cardiolog"],
    "Dermatology":              ["dermatolog", "cosmetolog"],
    "Paediatrics":              ["paediatric", "pediatric"],
    "Orthopaedics":             ["orthopaed", "orthoped"],
    "Obstetrics & Gynaecology": ["obstetric", "gynaecol", "gynecol"],
    "Internal Medicine":        ["internal medicine"],
    "Family Medicine":          ["family medicine"],
    "General Medicine":         ["general medicine", "general practitioner"],
    "General Surgery":          ["general surgeon", "general surgery"],
    "Physiotherapy":            ["physiotherap"],
    "ENT":                      ["ent", "otolaryngol", "ear, nose"],
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
    "General Practice":         ["general practice", "general practitioner"],
}


def _infer_speciality(title: str) -> str:
    t = title.lower()
    for spec, keywords in _SPEC_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return spec
    return ""


# ── Summary ───────────────────────────────────────────────────

def _print_summary(conn: sqlite3.Connection) -> None:
    tables = [
        "branches", "specialities", "branch_specialities",
        "doctors", "doctor_branches", "doctor_schedules",
        "insurance_providers", "branch_insurance", "health_packages",
    ]
    log.info("═" * 50)
    log.info("  DATABASE SUMMARY")
    for t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        log.info("  %-30s  %d rows", t, count)
    log.info("═" * 50)


# ── Entry point ───────────────────────────────────────────────

def run() -> None:
    log.info("═" * 50)
    log.info("  HealthHub DB Loader")
    log.info("  Input: %s", STRUCTURED_FILE)
    log.info("  DB:    %s", DB_PATH)
    log.info("═" * 50)

    # Load source data
    data = _load_data()
    log.info("  Source: %d doctors, %d branches, %d packages, %d text_content",
             len(data.get("doctors", [])),
             len(data.get("branches", [])),
             len(data.get("packages", [])),
             len(data.get("text_content", [])))

    # Connect and wipe
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Drop all tables for clean rebuild
    for t in ["doctor_schedules", "branch_insurance", "branch_specialities",
              "doctor_branches", "doctors", "branches", "specialities",
              "insurance_providers", "health_packages"]:
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.commit()

    conn.executescript(SCHEMA)
    conn.commit()

    log.info("━━ Branches ━━")
    load_branches(conn, data)

    log.info("━━ Specialities ━━")
    load_specialities(conn, data)

    log.info("━━ Branch ↔ Speciality ━━")
    load_branch_specialities(conn, data)

    log.info("━━ Doctors ━━")
    load_doctors(conn, data)

    log.info("━━ Doctor schedules (simulated) ━━")
    simulate_schedules(conn)

    log.info("━━ Insurance ━━")
    load_insurance(conn, data)

    log.info("━━ Branch insurance (simulated) ━━")
    simulate_branch_insurance(conn)

    log.info("━━ Health packages ━━")
    load_packages(conn, data)

    _print_summary(conn)
    conn.close()
    log.info("🎉 Database ready: %s", DB_PATH)


if __name__ == "__main__":
    run()