"""
rag/sql_retriever.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All structured SQL queries against healthhub.db.

One function per intent type that needs structured data.
Returns list of dicts — router combines with vector results.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

log = logging.getLogger(__name__)


def _conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "Run: python -m ingestion.db_loader"
        )
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn, sql: str, params: tuple = ()) -> list[dict]:
    """Execute SQL, return list of plain dicts."""
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log.error("SQL error: %s | params: %s | err: %s", sql[:80], params, e)
        return []


def _fuzzy_branch(branch_hint: Optional[str]) -> Optional[str]:
    """Convert extracted entity like 'JVC' → '%JVC%' for LIKE query."""
    if not branch_hint:
        return None
    return f"%{branch_hint}%"


# ── Branch queries ────────────────────────────────────────────

def get_branch_info(branch_hint: Optional[str] = None) -> list[dict]:
    """
    Get branch details: name, address, phone, hours, parking, maps_url.
    If branch_hint given → filter to matching branch.
    Otherwise → return all 12 branches.
    """
    with _conn() as conn:
        if branch_hint:
            return _rows(conn, """
                SELECT name, url, description, address, phone,
                       hours, parking, maps_url, specialities_json
                FROM branches
                WHERE name LIKE ?
                ORDER BY name
            """, (_fuzzy_branch(branch_hint),))
        return _rows(conn, """
            SELECT name, url, description, address, phone,
                   hours, parking, maps_url, specialities_json
            FROM branches
            ORDER BY name
        """)


def get_branch_specialities(branch_hint: Optional[str] = None) -> list[dict]:
    """Which specialities are available at which branch(es)."""
    with _conn() as conn:
        if branch_hint:
            return _rows(conn, """
                SELECT b.name AS branch_name, s.name AS speciality_name
                FROM branches b
                JOIN branch_specialities bs ON b.id = bs.branch_id
                JOIN specialities s         ON s.id = bs.speciality_id
                WHERE b.name LIKE ?
                ORDER BY s.name
            """, (_fuzzy_branch(branch_hint),))
        return _rows(conn, """
            SELECT b.name AS branch_name, s.name AS speciality_name
            FROM branches b
            JOIN branch_specialities bs ON b.id = bs.branch_id
            JOIN specialities s         ON s.id = bs.speciality_id
            ORDER BY b.name, s.name
        """)


# ── Doctor queries ────────────────────────────────────────────

def get_doctors(branch_hint: Optional[str] = None,
                speciality_hint: Optional[str] = None,
                doctor_name_hint: Optional[str] = None) -> list[dict]:
    """
    Flexible doctor lookup. Filters by any combination of
    branch, speciality, or name. All optional.
    """
    clauses = []
    params  = []

    if branch_hint:
        clauses.append("b.name LIKE ?")
        params.append(_fuzzy_branch(branch_hint))

    if speciality_hint:
        clauses.append("(d.speciality LIKE ? OR d.title LIKE ?)")
        params.extend([f"%{speciality_hint}%", f"%{speciality_hint}%"])

    if doctor_name_hint:
        clauses.append("d.name LIKE ?")
        params.append(f"%{doctor_name_hint}%")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    sql = f"""
        SELECT DISTINCT
            d.name, d.title, d.speciality,
            d.experience_years, d.nationality,
            d.languages_json, d.about, d.url,
            GROUP_CONCAT(b.name, ' | ') AS branches
        FROM doctors d
        LEFT JOIN doctor_branches db2 ON d.id = db2.doctor_id
        LEFT JOIN branches b          ON b.id = db2.branch_id
        {where}
        GROUP BY d.id
        ORDER BY d.name
        LIMIT 20
    """

    with _conn() as conn:
        rows = _rows(conn, sql, tuple(params))
        # Parse languages JSON
        for r in rows:
            try:
                r["languages"] = json.loads(r.get("languages_json", "[]"))
            except Exception:
                r["languages"] = []
        return rows


def get_doctor_schedule(doctor_name_hint: str,
                        branch_hint: Optional[str] = None,
                        day_hint: Optional[str] = None) -> list[dict]:
    """
    Get a doctor's availability schedule.
    SIMULATED data — clearly marked in results.
    """
    clauses = ["d.name LIKE ?"]
    params  = [f"%{doctor_name_hint}%"]

    if branch_hint:
        clauses.append("b.name LIKE ?")
        params.append(_fuzzy_branch(branch_hint))

    if day_hint:
        clauses.append("ds.day_of_week LIKE ?")
        params.append(f"%{day_hint}%")

    where = "WHERE " + " AND ".join(clauses)

    sql = f"""
        SELECT
            d.name AS doctor_name,
            d.title,
            b.name AS branch_name,
            ds.day_of_week,
            ds.slot_start,
            ds.slot_end,
            ds.is_simulated
        FROM doctor_schedules ds
        JOIN doctors  d ON d.id = ds.doctor_id
        JOIN branches b ON b.id = ds.branch_id
        {where}
        ORDER BY
            CASE ds.day_of_week
                WHEN 'Sunday'    THEN 1
                WHEN 'Monday'    THEN 2
                WHEN 'Tuesday'   THEN 3
                WHEN 'Wednesday' THEN 4
                WHEN 'Thursday'  THEN 5
                WHEN 'Friday'    THEN 6
                WHEN 'Saturday'  THEN 7
            END,
            ds.slot_start
    """

    with _conn() as conn:
        return _rows(conn, sql, tuple(params))


# ── Insurance queries ─────────────────────────────────────────

def get_insurance(insurance_hint: Optional[str] = None,
                  branch_hint: Optional[str] = None) -> list[dict]:
    """
    Check which insurance providers are accepted.
    If branch_hint: show acceptance for that branch specifically.
    Otherwise: show all accepted providers network-wide.
    SIMULATED branch-level data.
    """
    with _conn() as conn:
        if branch_hint and insurance_hint:
            # Specific: does branch X accept insurer Y?
            return _rows(conn, """
                SELECT
                    b.name AS branch_name,
                    ip.name AS insurance_name,
                    bi.is_simulated
                FROM branch_insurance bi
                JOIN branches          b  ON b.id  = bi.branch_id
                JOIN insurance_providers ip ON ip.id = bi.insurance_id
                WHERE b.name  LIKE ?
                  AND ip.name LIKE ?
            """, (_fuzzy_branch(branch_hint), f"%{insurance_hint}%"))

        if insurance_hint:
            # Which branches accept this insurer?
            return _rows(conn, """
                SELECT
                    b.name AS branch_name,
                    ip.name AS insurance_name,
                    bi.is_simulated
                FROM branch_insurance bi
                JOIN branches          b  ON b.id  = bi.branch_id
                JOIN insurance_providers ip ON ip.id = bi.insurance_id
                WHERE ip.name LIKE ?
                ORDER BY b.name
            """, (f"%{insurance_hint}%",))

        # All accepted providers
        return _rows(conn, """
            SELECT DISTINCT name AS insurance_name
            FROM insurance_providers
            ORDER BY name
        """)


# ── Package queries ───────────────────────────────────────────

def get_packages(name_hint: Optional[str] = None) -> list[dict]:
    """Get health packages, optionally filtered by name."""
    with _conn() as conn:
        if name_hint:
            return _rows(conn, """
                SELECT name, price_text, category, description, url
                FROM health_packages
                WHERE name LIKE ?
                ORDER BY name
            """, (f"%{name_hint}%",))
        return _rows(conn, """
            SELECT name, price_text, category, description, url
            FROM health_packages
            ORDER BY name
        """)


# ── Speciality queries ────────────────────────────────────────

def get_speciality_branches(speciality_hint: str) -> list[dict]:
    """Which branches offer a given speciality?"""
    with _conn() as conn:
        return _rows(conn, """
            SELECT DISTINCT b.name AS branch_name, b.phone, b.hours
            FROM branches b
            JOIN branch_specialities bs ON b.id = bs.branch_id
            JOIN specialities s         ON s.id = bs.speciality_id
            WHERE s.name LIKE ?
            ORDER BY b.name
        """, (f"%{speciality_hint}%",))
