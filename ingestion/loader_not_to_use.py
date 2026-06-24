import sqlite3
import json
from pathlib import Path

# ==========================================
# 1. DATABASE SETUP & SCHEMA DEFINITION
# ==========================================

def setup_database(db_path: str) -> sqlite3.Connection:
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enforce foreign key constraints in SQLite
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Drop existing tables to ensure a clean slate on every run
    tables = ["doctor_branches", "doctors", "branches", "insurance", "packages"]
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table};")

    # Create Tables
    cursor.executescript("""
        CREATE TABLE branches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            url TEXT,
            overview TEXT,
            specialities JSON
        );

        CREATE TABLE doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            url TEXT,
            title TEXT,
            experience_years INTEGER,
            nationality TEXT,
            languages JSON,
            expertise JSON,
            about TEXT
        );

        -- Critical Junction Table linking Doctors to Branches
        CREATE TABLE doctor_branches (
            doctor_id INTEGER,
            branch_id INTEGER,
            PRIMARY KEY (doctor_id, branch_id),
            FOREIGN KEY (doctor_id) REFERENCES doctors (id) ON DELETE CASCADE,
            FOREIGN KEY (branch_id) REFERENCES branches (id) ON DELETE CASCADE
        );

        CREATE TABLE insurance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            url TEXT,
            networks JSON
        );

        CREATE TABLE packages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            url TEXT,
            price TEXT,
            category TEXT,
            inclusions JSON
        );
    """)
    
    conn.commit()
    return conn

# ==========================================
# 2. DATA INGESTION & RELATIONSHIP MAPPING
# ==========================================

def load_data(conn: sqlite3.Connection, json_path: str):
    cursor = conn.cursor()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("[*] Beginning Data Ingestion into SQLite...")

    # 1. Load Branches First (Needed for Foreign Keys)
    print(f"    -> Loading {len(data.get('branches', []))} Branches...")
    branch_map = {} # Maps branch name to its ID for fast lookup
    for b in data.get('branches', []):
        cursor.execute("""
            INSERT INTO branches (name, url, overview, specialities)
            VALUES (?, ?, ?, ?)
        """, (b['name'], b['url'], b['overview'], json.dumps(b['specialities'])))
        branch_map[b['name'].lower()] = cursor.lastrowid

    # 2. Load Doctors & Map to Branches
    print(f"    -> Loading {len(data.get('doctors', []))} Doctors and establishing facility links...")
    for d in data.get('doctors', []):
        cursor.execute("""
            INSERT INTO doctors (name, url, title, experience_years, nationality, languages, expertise, about)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            d['name'], d['url'], d['title'], d['experience_years'], 
            d['nationality'], json.dumps(d['languages']), 
            json.dumps(d['expertise']), d['about']
        ))
        
        doctor_id = cursor.lastrowid

        # Resolve the string-based clinic names to actual branch IDs
        for clinic_name in d.get('clinics', []):
            clinic_lower = clinic_name.lower()
            branch_id = branch_map.get(clinic_lower)
            
            # Fuzzy fallback: If exact match fails, look for the clinic name inside the branch names
            if not branch_id:
                for b_name, b_id in branch_map.items():
                    if b_name in clinic_lower or clinic_lower in b_name:
                        branch_id = b_id
                        break
            
            if branch_id:
                # Insert relational link, ignoring duplicates if they exist
                cursor.execute("""
                    INSERT OR IGNORE INTO doctor_branches (doctor_id, branch_id)
                    VALUES (?, ?)
                """, (doctor_id, branch_id))

    # 3. Load Insurance
    print(f"    -> Loading {len(data.get('insurance', []))} Insurance Providers...")
    for i in data.get('insurance', []):
        cursor.execute("""
            INSERT OR IGNORE INTO insurance (name, url, networks)
            VALUES (?, ?, ?)
        """, (i['title'], i['url'], json.dumps(i['accepted_networks'])))

    # 4. Load Packages
    print(f"    -> Loading {len(data.get('packages', []))} Health Packages...")
    for p in data.get('packages', []):
        cursor.execute("""
            INSERT INTO packages (name, url, price, category, inclusions)
            VALUES (?, ?, ?, ?, ?)
        """, (p['package_name'], p['url'], p['price'], p['category'], json.dumps(p['inclusions'])))

    conn.commit()
    print("[✓] Relational database fully populated and linked.")

# ==========================================
# 3. EXECUTION
# ==========================================

if __name__ == "__main__":
    INPUT_JSON = "./data/structured/structured_data.json"
    OUTPUT_DB = "./data/db/healthhub.db"
    
    if not Path(INPUT_JSON).exists():
        print(f"[!] Error: Could not find structured data at {INPUT_JSON}")
    else:
        db_conn = setup_database(OUTPUT_DB)
        load_data(db_conn, INPUT_JSON)
        db_conn.close()
        print(f"\n[✓] Database ready at: {OUTPUT_DB}")