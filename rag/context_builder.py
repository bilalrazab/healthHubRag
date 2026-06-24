"""
rag/context_builder.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Formats all retrieval results (SQL + vector) into a
single context string for injection into the LLM prompt.

Design principles:
  - SQL results first (structured facts, highest precision)
  - Vector results second (semantic context)
  - Clear section separators so Claude knows data origin
  - Simulated data flagged explicitly
  - Never more than ~1500 words total in context
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
from typing import Optional


def _section(title: str, content: str) -> str:
    return f"### {title}\n{content.strip()}"


# ── SQL result formatters ─────────────────────────────────────

def format_branches(rows: list[dict]) -> str:
    if not rows:
        return ""
    parts = []
    for b in rows:
        lines = [f"**{b.get('name', '')}**"]
        if b.get("address"):
            lines.append(f"Address: {b['address']}")
        if b.get("phone"):
            lines.append(f"Phone: {b['phone']}")
        if b.get("hours"):
            lines.append(f"Hours: {b['hours']}")
        if b.get("parking"):
            lines.append(f"Parking: {b['parking']}")
        if b.get("maps_url"):
            lines.append(f"Directions: {b['maps_url']}")
        if b.get("description"):
            lines.append(b["description"][:200])
        try:
            specs = json.loads(b.get("specialities_json", "[]"))
            if specs:
                lines.append(f"Specialities: {', '.join(specs)}")
        except Exception:
            pass
        parts.append("\n".join(lines))
    return _section("Clinic Information", "\n\n".join(parts))


def format_doctors(rows: list[dict]) -> str:
    if not rows:
        return ""
    parts = []
    for d in rows:
        lines = [f"**{d.get('name', '')}**"]
        if d.get("title"):
            lines.append(f"Title: {d['title']}")
        if d.get("speciality"):
            lines.append(f"Speciality: {d['speciality']}")
        if d.get("experience_years"):
            lines.append(f"Experience: {d['experience_years']} years")
        if d.get("nationality"):
            lines.append(f"Nationality: {d['nationality']}")
        langs = d.get("languages") or []
        if isinstance(langs, str):
            try:
                langs = json.loads(langs)
            except Exception:
                langs = []
        if langs:
            lines.append(f"Languages: {', '.join(langs)}")
        if d.get("branches"):
            lines.append(f"Available at: {d['branches']}")
        if d.get("about"):
            lines.append(f"About: {d['about'][:200]}")
        parts.append("\n".join(lines))
    return _section("Doctors", "\n\n".join(parts))


def format_schedules(rows: list[dict]) -> str:
    if not rows:
        return ""
    parts = []
    doctor = rows[0].get("doctor_name", "Doctor")
    simulated = rows[0].get("is_simulated", 1)
    note = " *(Note: schedule is indicative — confirm when booking)*" if simulated else ""

    by_branch: dict[str, list] = {}
    for r in rows:
        branch = r.get("branch_name", "")
        by_branch.setdefault(branch, []).append(r)

    for branch, slots in by_branch.items():
        by_day: dict[str, list] = {}
        for s in slots:
            by_day.setdefault(s["day_of_week"], []).append(
                f"{s['slot_start']}–{s['slot_end']}"
            )
        schedule_lines = [f"{day}: {', '.join(times)}"
                          for day, times in by_day.items()]
        parts.append(
            f"**{doctor}** at {branch}{note}\n" + "\n".join(schedule_lines)
        )

    return _section("Doctor Availability", "\n\n".join(parts))


def format_insurance(rows: list[dict]) -> str:
    if not rows:
        return ""
    # Check if it's a name-only list (network-wide query)
    if "insurance_name" in rows[0] and "branch_name" not in rows[0]:
        names = [r["insurance_name"] for r in rows]
        return _section(
            "Accepted Insurance Providers",
            "HealthHub accepts the following insurance providers:\n"
            + ", ".join(names)
        )
    # Branch-specific
    parts = []
    for r in rows:
        sim_note = " *(verify when booking)*" if r.get("is_simulated") else ""
        parts.append(
            f"- {r['insurance_name']} at {r['branch_name']}{sim_note}"
        )
    return _section("Insurance Coverage", "\n".join(parts))


def format_packages(rows: list[dict]) -> str:
    if not rows:
        return ""
    parts = []
    for p in rows:
        lines = [f"**{p.get('name', '')}**"]
        if p.get("price_text"):
            lines.append(f"Price: AED {p['price_text']}")
        if p.get("category"):
            lines.append(f"Category: {p['category']}")
        if p.get("description"):
            lines.append(p["description"][:300])
        parts.append("\n".join(lines))
    return _section("Health Packages", "\n\n".join(parts))


def format_speciality_branches(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = [f"- {r['branch_name']} (Phone: {r.get('phone','N/A')}, Hours: {r.get('hours','N/A')})"
             for r in rows]
    return _section("Branches Offering This Speciality", "\n".join(lines))


# ── Vector result formatter ───────────────────────────────────

def format_vector_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_type", "")
        title  = chunk.get("title", "")
        header = f"[{i}] {title} ({source})" if title else f"[{i}] {source}"
        parts.append(f"{header}\n{chunk['text']}")
    return _section("Additional Information", "\n\n---\n\n".join(parts))


# ── Master context builder ────────────────────────────────────

def build(
    branch_rows:    Optional[list[dict]] = None,
    doctor_rows:    Optional[list[dict]] = None,
    schedule_rows:  Optional[list[dict]] = None,
    insurance_rows: Optional[list[dict]] = None,
    package_rows:   Optional[list[dict]] = None,
    spec_branch_rows: Optional[list[dict]] = None,
    vector_chunks:  Optional[list[dict]] = None,
) -> str:
    """
    Assemble all retrieval results into one context string.
    SQL results come first (higher precision), then vector.
    Empty sections are omitted.
    """
    sections = []

    if branch_rows:
        sections.append(format_branches(branch_rows))
    if doctor_rows:
        sections.append(format_doctors(doctor_rows))
    if schedule_rows:
        sections.append(format_schedules(schedule_rows))
    if insurance_rows:
        sections.append(format_insurance(insurance_rows))
    if package_rows:
        sections.append(format_packages(package_rows))
    if spec_branch_rows:
        sections.append(format_speciality_branches(spec_branch_rows))
    if vector_chunks:
        sections.append(format_vector_chunks(vector_chunks))

    sections = [s for s in sections if s.strip()]

    if not sections:
        return ""

    return "\n\n".join(sections)
