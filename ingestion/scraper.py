"""
ingestion/scraper.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HealthHub by Al-Futtaim — Firecrawl Scraper

Why Firecrawl:
  The site is Elementor WordPress. All content is JS-injected
  after page load. httpx + BeautifulSoup gets an empty HTML
  shell. Firecrawl runs a headless browser, waits for JS,
  then returns clean markdown. Problem solved.

Run once:
    python -m ingestion.scraper
    python -m ingestion.scraper --section branches
    python -m ingestion.scraper --section doctors
    python -m ingestion.scraper --section specialities
    python -m ingestion.scraper --section packages
    python -m ingestion.scraper --section general
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

from firecrawl import FirecrawlApp

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FIRECRAWL_API_KEY, CLINIC_URL, SCRAPE_DELAY,
    BRANCH_SLUGS, SPECIALITY_SLUGS, GENERAL_PAGES,
    INSURANCE_PROVIDERS, RAW_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scraper")

app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

SCRAPE_OPTS = {
    "formats": ["markdown"],
    "onlyMainContent": True,
    "waitFor": 2000,
    "timeout": 30000,
}


def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _scrape_url(url: str, retries: int = 3) -> dict | None:
    for attempt in range(1, retries + 1):
        try:
            result = app.scrape_url(url, params=SCRAPE_OPTS)
            markdown = result.get("markdown", "")
            metadata = result.get("metadata", {})
            if not markdown or len(markdown) < 100:
                log.warning("Short response (%d chars) for %s", len(markdown), url)
                if attempt < retries:
                    time.sleep(3 * attempt)
                    continue
                return None
            return {"url": url, "markdown": markdown, "metadata": metadata}
        except Exception as e:
            log.warning("Firecrawl error on %s (attempt %d): %s", url, attempt, e)
            if attempt < retries:
                wait = 3 * (2 ** (attempt - 1))
                log.info("Retrying in %ds...", wait)
                time.sleep(wait)
    log.error("Permanently failed: %s", url)
    return None


def scrape_branches() -> list[dict]:
    log.info("━━ BRANCHES (%d) ━━", len(BRANCH_SLUGS))
    results = []
    for slug in BRANCH_SLUGS:
        url = f"{CLINIC_URL}/clinics/{slug}/"
        log.info("  → %s", url)
        data = _scrape_url(url)
        if data is None:
            results.append({"slug": slug, "url": url, "markdown": "", "metadata": {}, "scraped_ok": False})
        else:
            data["slug"] = slug
            data["scraped_ok"] = True
            results.append(data)
            log.info("  ✓ %-45s  %d chars", slug, len(data["markdown"]))
            _save(data, RAW_DIR / "branches" / f"{slug}.json")
        time.sleep(SCRAPE_DELAY)
    log.info("✅ Branches: %d/%d ok\n", sum(r["scraped_ok"] for r in results), len(results))
    return results


def collect_doctor_urls() -> list[str]:
    urls = set()
    branch_dir = RAW_DIR / "branches"
    if not branch_dir.exists():
        log.warning("No branch data. Run --section branches first.")
        return []
    for jf in sorted(branch_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            found = re.findall(
                r'https://www\.healthhubalfuttaim\.com/doctor/[a-z0-9\-]+/?',
                data.get("markdown", "")
            )
            for u in found:
                urls.add(u.split("#")[0].rstrip("/") + "/")
        except Exception as e:
            log.warning("Error reading %s: %s", jf, e)
    result = sorted(urls)
    log.info("  Collected %d unique doctor URLs from branch pages", len(result))
    return result


def scrape_doctors() -> list[dict]:
    doctor_urls = collect_doctor_urls()
    if not doctor_urls:
        log.warning("No doctor URLs found. Trying /doctor/ listing page...")
        listing = _scrape_url(f"{CLINIC_URL}/doctor/")
        time.sleep(SCRAPE_DELAY)
        if listing:
            found = re.findall(
                r'https://www\.healthhubalfuttaim\.com/doctor/[a-z0-9\-]+/?',
                listing["markdown"]
            )
            doctor_urls = sorted({u.split("#")[0].rstrip("/") + "/" for u in found})
            log.info("  Found %d doctor URLs from listing page", len(doctor_urls))

    log.info("━━ DOCTORS (%d) ━━", len(doctor_urls))
    results = []
    for url in doctor_urls:
        slug = url.rstrip("/").split("/doctor/")[-1]
        log.info("  → %s", slug)
        data = _scrape_url(url)
        if data is None:
            results.append({"slug": slug, "url": url, "markdown": "", "metadata": {}, "scraped_ok": False})
        else:
            data["slug"] = slug
            data["scraped_ok"] = True
            results.append(data)
            log.info("  ✓ %-45s  %d chars", slug, len(data["markdown"]))
            _save(data, RAW_DIR / "doctors" / f"{slug}.json")
        time.sleep(SCRAPE_DELAY)
    log.info("✅ Doctors: %d/%d ok\n", sum(r["scraped_ok"] for r in results), len(results))
    return results


def scrape_specialities() -> list[dict]:
    log.info("━━ SPECIALITIES (%d) ━━", len(SPECIALITY_SLUGS))
    results = []
    for slug in SPECIALITY_SLUGS:
        url = f"{CLINIC_URL}/specialties/{slug}/"
        log.info("  → %s", slug)
        data = _scrape_url(url)
        if data is None:
            results.append({"slug": slug, "url": url, "markdown": "", "metadata": {}, "scraped_ok": False})
        else:
            data["slug"] = slug
            data["scraped_ok"] = True
            results.append(data)
            log.info("  ✓ %-50s  %d chars", slug, len(data["markdown"]))
            _save(data, RAW_DIR / "specialities" / f"{slug}.json")
        time.sleep(SCRAPE_DELAY)
    log.info("✅ Specialities: %d/%d ok\n", sum(r["scraped_ok"] for r in results), len(results))
    return results


def scrape_packages() -> list[dict]:
    log.info("━━ HEALTH PACKAGES ━━")
    index_url = f"{CLINIC_URL}/health-packages/"
    log.info("  → Fetching package index")
    index_data = _scrape_url(index_url)
    time.sleep(SCRAPE_DELAY)
    results = []
    if index_data is None:
        log.error("Failed to fetch package index")
        return results
    _save({**index_data, "slug": "packages-index", "scraped_ok": True},
          RAW_DIR / "packages" / "packages-index.json")
    pkg_urls = sorted(set(re.findall(
        r'https://www\.healthhubalfuttaim\.com/packages/[a-z0-9\-]+/?',
        index_data["markdown"]
    )))
    log.info("  Found %d package URLs", len(pkg_urls))
    for url in pkg_urls:
        slug = url.rstrip("/").split("/")[-1]
        log.info("  → %s", slug)
        data = _scrape_url(url)
        if data is None:
            results.append({"slug": slug, "url": url, "markdown": "", "metadata": {}, "scraped_ok": False})
        else:
            data["slug"] = slug
            data["scraped_ok"] = True
            results.append(data)
            log.info("  ✓ %-45s  %d chars", slug, len(data["markdown"]))
            _save(data, RAW_DIR / "packages" / f"{slug}.json")
        time.sleep(SCRAPE_DELAY)
    log.info("✅ Packages: %d/%d ok\n", sum(r["scraped_ok"] for r in results), len(results))
    return results


def scrape_insurance() -> dict:
    log.info("━━ INSURANCE ━━")
    data = {"providers": INSURANCE_PROVIDERS, "scraped_ok": True}
    url = f"{CLINIC_URL}/insurance-provider/"
    page = _scrape_url(url)
    time.sleep(SCRAPE_DELAY)
    if page:
        data["page_markdown"] = page["markdown"]
        log.info("  ✓ Insurance page: %d chars", len(page["markdown"]))
    else:
        data["page_markdown"] = ""
    _save(data, RAW_DIR / "insurance" / "insurance_providers.json")
    log.info("✅ Insurance: %d providers saved\n", len(INSURANCE_PROVIDERS))
    return data


def scrape_general() -> list[dict]:
    log.info("━━ GENERAL PAGES (%d) ━━", len(GENERAL_PAGES))
    results = []
    for key, path in GENERAL_PAGES:
        url = CLINIC_URL + path
        log.info("  → %s  (%s)", key, url)
        data = _scrape_url(url)
        if data is None:
            results.append({"key": key, "url": url, "markdown": "", "metadata": {}, "scraped_ok": False})
        else:
            data["key"] = key
            data["scraped_ok"] = True
            results.append(data)
            log.info("  ✓ %-20s  %d chars", key, len(data["markdown"]))
            _save(data, RAW_DIR / "general" / f"{key}.json")
        time.sleep(SCRAPE_DELAY)
    log.info("✅ General: %d/%d ok\n", sum(r["scraped_ok"] for r in results), len(results))
    return results


def save_summary(branches, doctors, specialities, insurance, packages, general):
    def ok(lst): return sum(r.get("scraped_ok", False) for r in lst)
    summary = {
        "branches":     {"total": len(branches),     "ok": ok(branches)},
        "doctors":      {"total": len(doctors),      "ok": ok(doctors)},
        "specialities": {"total": len(specialities), "ok": ok(specialities)},
        "insurance":    {"total": len(INSURANCE_PROVIDERS)},
        "packages":     {"total": len(packages),     "ok": ok(packages)},
        "general":      {"total": len(general),      "ok": ok(general)},
    }
    p = RAW_DIR / "scrape_summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("═" * 56)
    log.info("  SCRAPE SUMMARY")
    for k, v in summary.items():
        if "ok" in v:
            log.info("  %-15s %d / %d", k.capitalize() + ":", v["ok"], v["total"])
        else:
            log.info("  %-15s %d", k.capitalize() + ":", v["total"])
    log.info("═" * 56)


def run(section: str = "all") -> None:
    log.info("═" * 56)
    log.info("  HealthHub Scraper — Firecrawl — section: %s", section)
    log.info("  Output: %s", RAW_DIR)
    log.info("═" * 56)
    for d in ["branches", "doctors", "specialities", "insurance", "packages", "general"]:
        (RAW_DIR / d).mkdir(parents=True, exist_ok=True)

    branches = doctors = specialities = packages = general = []
    insurance = {}

    if section in ("all", "branches"):     branches     = scrape_branches()
    if section in ("all", "specialities"): specialities = scrape_specialities()
    if section in ("all", "doctors"):      doctors      = scrape_doctors()
    if section in ("all", "insurance"):    insurance    = scrape_insurance()
    if section in ("all", "packages"):     packages     = scrape_packages()
    if section in ("all", "general"):      general      = scrape_general()

    if section == "all":
        save_summary(branches, doctors, specialities,
                     [insurance], packages, general)
    log.info("🎉 Done. Raw files in: %s", RAW_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HealthHub Firecrawl Scraper")
    parser.add_argument(
        "--section",
        choices=["all", "branches", "doctors", "specialities",
                 "insurance", "packages", "general"],
        default="all",
    )
    args = parser.parse_args()
    run(section=args.section)
