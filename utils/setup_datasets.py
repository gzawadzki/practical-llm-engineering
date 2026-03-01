"""
setup_datasets.py
-----------------
Downloads and prepares ALL datasets used in the Practical LLM course (Modules 1-7).

Prerequisites
-------------
  pip install -r requirements.txt
  # For Kaggle datasets: configure ~/.kaggle/kaggle.json first
  #   https://www.kaggle.com/docs/api#authentication

Run
---
  python setup_datasets.py
  python setup_datasets.py --module 1        # only M1 datasets
  python setup_datasets.py --module 1 2 3   # specific modules
  python setup_datasets.py --skip-kaggle    # skip Kaggle datasets (no API key)
"""

import argparse
import os
import sys
import zipfile
import subprocess
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Download all course datasets")
parser.add_argument("--module", nargs="*", type=int, help="Modules to download (e.g. 1 2 3)")
parser.add_argument("--skip-kaggle", action="store_true", help="Skip Kaggle datasets")
parser.add_argument("--data-dir", default="data", help="Root data directory (default: data)")
args, _ = parser.parse_known_args()

MODULES  = set(args.module) if args.module else set(range(1, 10))
SKIP_KAGGLE = args.skip_kaggle
DATA_DIR = Path(args.data_dir)
DATA_DIR.mkdir(exist_ok=True)

print(f"\n📁 Data directory : {DATA_DIR.resolve()}")
print(f"   Modules        : {sorted(MODULES)}")
print(f"   Skip Kaggle    : {SKIP_KAGGLE}\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: str) -> bool:
    """Run shell command. Returns True on success."""
    print(f"  ▶ {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def unzip_all_in(folder: Path):
    for file in folder.glob("*.zip"):
        print(f"  📦 Unzipping {file.name}")
        with zipfile.ZipFile(file, "r") as zf:
            zf.extractall(folder)
        file.unlink()


def hf_load(dataset_id, config=None, split="train", select=None, trust_remote_code=False):
    """Load a HuggingFace dataset safely, return None on failure."""
    from datasets import load_dataset
    try:
        kwargs = dict(split=split, trust_remote_code=trust_remote_code)
        if config:
            ds = load_dataset(dataset_id, config, **kwargs)
        else:
            ds = load_dataset(dataset_id, **kwargs)
        if select is not None:
            ds = ds.select(range(select))
        return ds
    except Exception as e:
        print(f"  ⚠️  Could not load {dataset_id}: {e}")
        return None


def save_hf(ds, path: Path, name: str = "data.csv"):
    if ds is None:
        return
    path.mkdir(parents=True, exist_ok=True)
    df = ds.to_pandas()
    df.to_csv(path / name, index=False)
    print(f"  ✅ Saved {len(df)} rows → {path / name}")


# ── Module 1 — E-commerce & Reviews ──────────────────────────────────────────

def download_m1():
    print("\n══════════ Module 1: E-commerce & Customer Reviews ══════════")

    # Amazon Reviews 2023 (Electronics)
    print("\n[1/2] Amazon Product Reviews (Electronics) — HuggingFace")
    amazon_dir = DATA_DIR / "amazon_reviews"
    ds = hf_load(
        "McAuley-Lab/Amazon-Reviews-2023",
        config="raw_review_Electronics",
        split="full",
        select=10_000,
        trust_remote_code=True,
    )
    save_hf(ds, amazon_dir, "amazon_electronics_10k.csv")

    # Yelp Reviews — Kaggle
    print("\n[2/2] Yelp Open Dataset — Kaggle")
    yelp_dir = DATA_DIR / "yelp"
    yelp_dir.mkdir(exist_ok=True)
    if SKIP_KAGGLE:
        print("  ⏭️  Skipped (--skip-kaggle)")
    else:
        ok = run(f"kaggle datasets download -d yelp-dataset -p {yelp_dir} --unzip")
        if ok:
            # Sample 10k rows from the large review file
            import pandas as pd, json
            yelp_json = yelp_dir / "yelp_academic_dataset_review.json"
            if yelp_json.exists():
                records = []
                with open(yelp_json) as f:
                    for i, line in enumerate(f):
                        if i >= 10_000:
                            break
                        records.append(json.loads(line))
                pd.DataFrame(records).to_csv(yelp_dir / "yelp_review_sample.csv", index=False)
                print(f"  ✅ Saved 10k Yelp reviews → {yelp_dir/'yelp_review_sample.csv'}")
            else:
                print("  ⚠️  yelp_academic_dataset_review.json not found after download")
        else:
            print("  ⚠️  Kaggle download failed — check ~/.kaggle/kaggle.json")


# ── Module 2 — Legal & Contracts ──────────────────────────────────────────────

def download_m2():
    print("\n══════════ Module 2: Legal & Contracts ══════════")

    # SEC EDGAR 10-K filings
    print("\n[1/2] SEC EDGAR 10-K filings")
    sec_dir = DATA_DIR / "sec_10k"
    sec_dir.mkdir(exist_ok=True)
    try:
        from sec_edgar_downloader import Downloader
        dl = Downloader("MyCompany", "myemail@example.com", str(sec_dir))
        for ticker in ["AAPL", "MSFT", "AMZN"]:
            try:
                dl.get("10-K", ticker, limit=2)
                print(f"  ✅ 10-K for {ticker}")
            except Exception as e:
                print(f"  ⚠️  {ticker}: {e}")
    except ImportError:
        print("  ⚠️  sec-edgar-downloader not installed: pip install sec-edgar-downloader")

    # EUR-Lex legal documents
    print("\n[2/2] EUR-Lex legal documents — HuggingFace")
    eurlex_dir = DATA_DIR / "eurlex"
    ds = hf_load("rcadas/EUR-Lex", split="train", select=500)
    save_hf(ds, eurlex_dir, "eurlex_500.csv")


# ── Module 3 — Scientific Literature ──────────────────────────────────────────

def download_m3():
    print("\n══════════ Module 3: Scientific Literature (RAG) ══════════")

    # arXiv abstracts
    print("\n[1/3] arXiv abstracts 2021 — HuggingFace")
    arxiv_dir = DATA_DIR / "arxiv"
    ds = hf_load("gfissore/arxiv-abstracts-2021", split="train", select=5_000)
    save_hf(ds, arxiv_dir, "arxiv_abstracts_5k.csv")

    # PubMed QA
    print("\n[2/3] PubMed QA (labeled) — HuggingFace")
    pubmed_dir = DATA_DIR / "pubmed_qa"
    ds = hf_load("pubmed_qa", config="pqa_labeled", split="train")
    save_hf(ds, pubmed_dir, "pubmed_qa_labeled.csv")

    # Wikipedia (English)
    print("\n[3/3] Wikipedia (English) — HuggingFace (500 articles)")
    wiki_dir = DATA_DIR / "wikipedia"
    ds = hf_load("wikipedia", config="20220301.en", split="train", select=500)
    save_hf(ds, wiki_dir, "wikipedia_en_500.csv")

    # Polish Wikipedia (for 03d)
    print("\n[3b/3] Wikipedia (Polish) — HuggingFace (200 articles)")
    ds = hf_load("wikipedia", config="20230901.pl", split="train", select=200)
    save_hf(ds, wiki_dir, "wikipedia_pl_200.csv")


# ── Module 4 — IT / DevOps ────────────────────────────────────────────────────

def download_m4():
    print("\n══════════ Module 4: IT / DevOps & Incident Reports ══════════")

    print("\n[1/2] GitHub Issues small — HuggingFace")
    gh_dir = DATA_DIR / "github_issues"
    ds = hf_load("giganticode/github-issues-small", split="train")
    save_hf(ds, gh_dir, "github_issues.csv")

    print("\n[2/2] Stack Overflow Questions — Kaggle")
    so_dir = DATA_DIR / "stackoverflow"
    so_dir.mkdir(exist_ok=True)
    if SKIP_KAGGLE:
        print("  ⏭️  Skipped (--skip-kaggle)")
    else:
        run(f"kaggle datasets download -d stackoverflow-questions -p {so_dir} --unzip")


# ── Module 5 — Healthcare / Clinical ─────────────────────────────────────────

def download_m5():
    print("\n══════════ Module 5: Healthcare / Clinical Notes ══════════")

    print("\n[1/2] MedQA (USMLE questions) — HuggingFace")
    medqa_dir = DATA_DIR / "medqa"
    ds = hf_load(
        "bigbio/med_qa",
        config="med_qa_en_bigbio_qa",
        split="train",
        select=500,
        trust_remote_code=True,
    )
    save_hf(ds, medqa_dir, "medqa_500.csv")

    print("\n[2/2] Heart Disease UCI — Kaggle")
    heart_dir = DATA_DIR / "heart_disease"
    heart_dir.mkdir(exist_ok=True)
    if SKIP_KAGGLE:
        print("  ⏭️  Skipped (--skip-kaggle)")
    else:
        run(f"kaggle datasets download -d cherngs/heart-disease-cleveland-uci -p {heart_dir} --unzip")


# ── Module 6 — Evaluation, Evals & LLMOps ────────────────────────────────────

def download_m6():
    print("\n══════════ Module 6: Evaluation, Evals & LLMOps ══════════")
    # M6 re-uses arXiv data (M3), GitHub issues (M4), and triage cards (MC2).
    # Golden eval sets (golden_eval_arxiv.jsonl, triage_cards.jsonl) are generated
    # inside the notebooks — no separate dataset download is required.
    print("\n  ℹ️  Module 6 datasets come from M3, M4, and Mini-Capstone 2.")
    print("      Run modules 3–5 downloads first.")

    # Langfuse (06c) runs via Docker — no data download needed.
    print("\n  ℹ️  Langfuse (06c): self-hosted via Docker.")
    print("      git clone https://github.com/langfuse/langfuse && docker compose up -d")


# ── Module 7 — Banking CRM ────────────────────────────────────────────────────

def download_m7():
    print("\n══════════ Module 7: Banking CRM (Advanced 2026 + Capstone) ══════════")

    # CFPB Consumer Complaints — HuggingFace (preferred) or direct download
    print("\n[1/3] CFPB Consumer Complaints — HuggingFace")
    cfpb_dir = DATA_DIR / "cfpb_complaints"
    ds = hf_load("cfpb/us-consumer-finance-complaints", split="train", select=5_000)
    if ds is not None:
        save_hf(ds, cfpb_dir, "cfpb_complaints_5k.csv")
    else:
        print("  Trying direct download from CFPB…")
        cfpb_dir.mkdir(exist_ok=True)
        zip_path = cfpb_dir / "complaints.csv.zip"
        ok = run(f"curl -L -o {zip_path} https://files.consumerfinance.gov/ccdb/complaints.csv.zip")
        if ok:
            unzip_all_in(cfpb_dir)

    # Telco Customer Churn — Kaggle (banking churn proxy in 07c + Capstone)
    print("\n[2/3] Telco Customer Churn — Kaggle")
    telco_dir = DATA_DIR / "telco_churn"
    telco_dir.mkdir(exist_ok=True)
    if SKIP_KAGGLE:
        print("  ⏭️  Skipped (--skip-kaggle)")
    else:
        ok = run(f"kaggle datasets download -d blastchar/telco-customer-churn -p {telco_dir}")
        if ok:
            unzip_all_in(telco_dir)

    # Synthetic bank statements (07b) are generated in-notebook via ReportLab.
    print("\n[3/3] Synthetic bank statements (07b) — generated in-notebook via ReportLab.")
    print("      No download needed. Ensure reportlab is installed: pip install reportlab Pillow")



# ── Module 8 — Capstone Preparation ──────────────────────────────────────────

def download_m8():
    print("\n══════════ Module 8: Capstone Preparation ══════════")

    # CFPB Consumer Complaints — same source as M7, larger slice
    print("\n[1/3] CFPB Consumer Complaints (100 rows for M8) — HuggingFace")
    cfpb_dir = DATA_DIR / "cfpb_complaints"
    if (cfpb_dir / "cfpb_complaints_5k.csv").exists():
        print("  ✅ Already downloaded in Module 7 — reusing")
    else:
        ds = hf_load("cfpb/us-consumer-finance-complaints", split="train", select=100)
        save_hf(ds, cfpb_dir, "cfpb_complaints_100.csv")

    # Telco Customer Churn — same source as M7 (XGBoost churn model in M8)
    print("\n[2/3] Telco Customer Churn (XGBoost model) — Kaggle")
    telco_dir = DATA_DIR / "telco_churn"
    telco_dir.mkdir(exist_ok=True)
    if (telco_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv").exists():
        print("  ✅ Already downloaded in Module 7 — reusing")
    elif SKIP_KAGGLE:
        print("  ⏭️  Skipped (--skip-kaggle) — M8 will use synthetic fallback data")
    else:
        ok = run(f"kaggle datasets download -d blastchar/telco-customer-churn -p {telco_dir}")
        if ok:
            unzip_all_in(telco_dir)

    # Capstone output directory
    print("\n[3/3] Creating capstone output directory")
    cap_dir = DATA_DIR / "capstone"
    cap_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ Created {cap_dir} (will be populated by Module 8 notebooks)")


# ── Capstone — Banking CRM ────────────────────────────────────────────────────

def download_capstone():
    print("\n══════════ Capstone: Banking CRM Intelligence System ══════════")
    print("\n  ℹ️  Capstone datasets come from M7 and M8:")
    print("      • CFPB Consumer Complaints  → data/cfpb_complaints/")
    print("      • Telco Churn (XGBoost)     → data/telco_churn/")
    print("      • Synthetic bank statements → generated in-notebook (ReportLab)")
    print("      • Policy ChromaDB           → built in-notebook from LM Studio")
    print("      • M8 JSONL cache            → data/capstone/ (created by Module 8)")
    print("\n  ➡️  Run modules 7 and 8 downloads first.")

    # Ensure capstone dir exists
    cap_dir = DATA_DIR / "capstone"
    cap_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  ✅ {cap_dir} ready")




DOWNLOADERS = {
    1: download_m1,
    2: download_m2,
    3: download_m3,
    4: download_m4,
    5: download_m5,
    6: download_m6,
    7: download_m7,
    8: download_m8,
    9: download_capstone,  # Capstone = "Module 9"
}

try:
    from datasets import load_dataset  # noqa: F401
except ImportError:
    print("❌ 'datasets' library not found. Run: pip install -r requirements.txt")
    sys.exit(1)

errors = []
for module_num in sorted(MODULES):
    fn = DOWNLOADERS.get(module_num)
    if fn is None:
        print(f"\nℹ️  Module {module_num}: no separate download needed (uses previous module data).")
        continue
    try:
        fn()
    except Exception as e:
        print(f"\n❌ Module {module_num} failed: {e}")
        errors.append((module_num, e))

print("\n" + "═" * 60)
if errors:
    print(f"⚠️  Completed with {len(errors)} error(s):")
    for m, e in errors:
        print(f"   Module {m}: {e}")
else:
    print("🎉 ALL DATASETS DOWNLOADED SUCCESSFULLY")
print(f"\n📂 Location: {DATA_DIR.resolve()}")