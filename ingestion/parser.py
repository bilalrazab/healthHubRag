import os
import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# ==========================================
# 1. STRUCTURAL SCHEMAS (Pydantic Validation)
# ==========================================

class DoctorData(BaseModel):
    url: str
    name: str
    title: str
    experience_years: Optional[int] = None
    languages: List[str] = Field(default_factory=list)
    nationality: Optional[str] = None
    clinics: List[str] = Field(default_factory=list)
    about: str
    expertise: List[str] = Field(default_factory=list)

class BranchData(BaseModel):
    url: str
    name: str
    overview: str
    specialities: List[str] = Field(default_factory=list)

class InsuranceData(BaseModel):
    url: str
    title: str
    accepted_networks: List[str] = Field(default_factory=list)

class PackageData(BaseModel):
    url: str
    package_name: str
    price: Optional[str] = None
    category: Optional[str] = None
    inclusions: List[str] = Field(default_factory=list)

class TextContentData(BaseModel):
    url: str
    type: str  # 'speciality', 'article', 'news'
    title: str
    clean_body: str

class MasterStructuredOutput(BaseModel):
    doctors: List[DoctorData] = Field(default_factory=list)
    branches: List[BranchData] = Field(default_factory=list)
    insurance: List[InsuranceData] = Field(default_factory=list)
    packages: List[PackageData] = Field(default_factory=list)
    text_content: List[TextContentData] = Field(default_factory=list)


# ==========================================
# 2. TARGETED EXTRACTION STRATEGIES
# ==========================================

def parse_doctor(payload: Dict[str, Any]) -> DoctorData:
    markdown = payload.get("markdown", "")
    # Slice off dynamic booking tables or patient comments widgets instantly
    clean_text = re.split(r'#### Patient Stories|###### Book Appointment|###### Select Date', markdown)[0]
    
    name_match = re.search(r'#\s*(Dr\..+|Ms\..+|Prof\..+)', clean_text)
    name = name_match.group(1).strip() if name_match else "Unknown Doctor"
    
    title = ""
    if name_match:
        after_name = clean_text.split(name_match.group(0))[-1].strip()
        lines = [l.strip() for l in after_name.split('\n') if l.strip()]
        if lines and "no excerpt found" not in lines[0]:
            title = lines[0]

    exp_match = re.search(r'(\d+)\+?\s*years?\s*of\s*exp', clean_text, re.IGNORECASE)
    exp = int(exp_match.group(1)) if exp_match else None
    
    lang_match = re.search(r'Languages\s*([A-Za-z,\s]+)', clean_text)
    languages = [l.strip() for l in lang_match.group(1).replace('\n','').split(',')] if lang_match else []
    
    nat_match = re.search(r'Nationality\s*([A-Za-z\s]+)', clean_text)
    nationality = nat_match.group(1).strip() if nat_match else None
    
    clinics = [c.strip() for c in re.findall(r'\*\*(HealthHub[^*]+)\*\*', clean_text)]
    
    about_text = ""
    about_match = re.search(r'#### About\n+(.*?)(?=\n####|\Z)', clean_text, re.DOTALL)
    if about_match:
        about_text = about_match.group(1).strip()
        
    expertise = []
    exp_block = re.search(r'#### Expertise\n+(.*?)(?=\n####|\Z)', clean_text, re.DOTALL)
    if exp_block:
        expertise = [b.strip() for b in re.findall(r'-\s*(.+)', exp_block.group(1)) if b.strip()]

    return DoctorData(
        url=payload.get("url", ""), name=name, title=title, experience_years=exp,
        languages=languages, nationality=nationality, clinics=list(set(clinics)),
        about=about_text, expertise=expertise
    )

def parse_branch(payload: Dict[str, Any]) -> BranchData:
    markdown = payload.get("markdown", "")
    clean_text = re.split(r'## Our Doctors|Our Doctors', markdown)[0]
    
    name_match = re.search(r'#\s*(.+)', clean_text)
    name = name_match.group(1).strip() if name_match else "HealthHub Location"
    
    overview = ""
    if name_match:
        after_name = clean_text.split(name_match.group(0))[-1].strip()
        paragraphs = [p.strip() for p in after_name.split('\n\n') if p.strip() and not p.startswith('!')]
        if paragraphs:
            overview = paragraphs[0]
            
    specialities = []
    spec_match = re.search(r'## Specialities\n+(.*?)(?=\n##|\Z)', clean_text, re.DOTALL)
    if spec_match:
        items = re.findall(r'-\s*(?:!\[.*?\]\(.*?\))?\s*(.+)', spec_match.group(1))
        specialities = [i.strip() for i in items if i.strip()]
        
    return BranchData(url=payload.get("url", ""), name=name, overview=overview, specialities=list(set(specialities)))

def parse_insurance(payload: Dict[str, Any]) -> InsuranceData:
    markdown = payload.get("markdown", "")
    title_match = re.search(r'#\s*(.+)', markdown)
    title = title_match.group(1).strip() if title_match else "Insurance Providers"
    
    # Extract clean list patterns matching company bullet items
    networks = [n.strip() for n in re.findall(r'-\s*(.+)', markdown) if "http" not in n and len(n) < 100]
    return InsuranceData(url=payload.get("url", ""), title=title, accepted_networks=networks)

def parse_package(payload: Dict[str, Any]) -> PackageData:
    markdown = payload.get("markdown", "")
    title_match = re.search(r'#\s*(.+)', markdown)
    name = title_match.group(1).strip() if title_match else "Health Package"
    
    price_match = re.search(r'(?:AED|Price:?)\s*([\d,]+)', markdown, re.IGNORECASE)
    price = price_match.group(1).strip() if price_match else None
    
    inclusions = [i.strip() for i in re.findall(r'-\s*(.+)', markdown) if len(i) > 3 and "http" not in i]
    return PackageData(url=payload.get("url", ""), package_name=name, price=price, category="Health Screening", inclusions=inclusions)

def parse_generic_text(payload: Dict[str, Any], content_type: str) -> TextContentData:
    markdown = payload.get("markdown", "")
    title_match = re.search(r'#\s*(.+)', markdown)
    title = title_match.group(1).strip() if title_match else "Untitled Document"
    
    # Strip common header navigation words or cookie warnings if applicable
    clean_body = re.sub(r'(!\[.*?\]\(.*?\))|\[Read More\].*?\n', '', markdown, flags=re.MULTILINE).strip()
    return TextContentData(url=payload.get("url", ""), type=content_type, title=title, clean_body=clean_body)


# ==========================================
# 3. CORE ROUTING PIPELINE ENGINE
# ==========================================

def execute_pipeline(input_dir: str, output_file_path: str):
    root = Path(input_dir)
    master_store = MasterStructuredOutput()
    
    if not root.exists():
        print(f"[!] Target path does not exist: {root.resolve()}")
        return

    # Scan completely through all subfolders recursively
    all_json_files = list(root.rglob("*.json"))
    print(f"[*] Starting extraction across {len(all_json_files)} structural payloads...")

    for file_path in all_json_files:
        if file_path.name in ["memory_ccmf.json", "structured_data.json"]:
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            
            url = payload.get("url", "").lower()
            parent_folder = file_path.parent.name.lower()
            
            # Dynamic pipeline router mapping based on file locations & url path keywords
            if "doctor" in url or parent_folder == "doctor":
                master_store.doctors.append(parse_doctor(payload))
            elif "clinics" in url or parent_folder == "clinics":
                master_store.branches.append(parse_branch(payload))
            elif "insurance" in url or parent_folder == "insurance":
                master_store.insurance.append(parse_insurance(payload))
            elif any(k in url for k in ["package", "corporate"]) or parent_folder in ["packages", "corporate-packages"]:
                master_store.packages.append(parse_package(payload))
            elif "specialities" in url or parent_folder == "specialities":
                master_store.text_content.append(parse_generic_text(payload, "speciality"))
            elif "articles" in url or parent_folder == "articles":
                master_store.text_content.append(parse_generic_text(payload, "article"))
            elif "news" in url or parent_folder == "news":
                master_store.text_content.append(parse_generic_text(payload, "news"))
            else:
                # Catch-all baseline for root or unexpected folders
                master_store.text_content.append(parse_generic_text(payload, "general"))
                
        except Exception as e:
            print(f"[!] Processing Error inside target [{file_path.name}]: {str(e)}")

    # Write output cleanly
    out_path = Path(output_file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(master_store.model_dump(), out_f, indent=2, ensure_ascii=False)
        
    print(f"\n[✓] Pipeline Execution Successful!")
    print(f"    -> Parsed Doctors: {len(master_store.doctors)}")
    print(f"    -> Parsed Branches: {len(master_store.branches)}")
    print(f"    -> Parsed Insurances: {len(master_store.insurance)}")
    print(f"    -> Parsed Packages: {len(master_store.packages)}")
    print(f"    -> Parsed Articles/Specialties: {len(master_store.text_content)}")
    print(f"    -> Document location: {out_path.resolve()}")

if __name__ == "__main__":
    # Point this to your top level directory containing the crawled folders
    RAW_DATA_INPUT = "./data/raw"
    STRUCTURED_OUTPUT = "./data/structured/structured_data.json"
    
    execute_pipeline(RAW_DATA_INPUT, STRUCTURED_OUTPUT)