"""Bundle all deliverables into Final_Capstone_Project.zip.

Excludes secrets (.env), caches, and build artifacts. Includes .env.example.
Run:  python make_zip.py
"""
import zipfile
from pathlib import Path

ROOT = Path(".")
ZIP_NAME = "Final_Capstone_Project.zip"

INCLUDE = [
    "project.ipynb",
    "app.py",
    "requirements.txt",
    "README.md",
    "report.pdf",
    "report.pptx",
    ".env.example",
    ".gitignore",
    "build_notebook.py",
    "build_report.py",
    "build_report_pptx.py",
    "src/rag_pipeline.py",
    "data/rag_overview.md",
    "data/llm_concepts.md",
    "data/vector_databases.md",
    "assets/architecture.png",
]

missing = [f for f in INCLUDE if not (ROOT / f).exists()]
if missing:
    raise SystemExit(f"Cannot build zip, missing files: {missing}")

# Sanity: never ship the real secrets file.
assert ".env" not in INCLUDE

with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in INCLUDE:
        zf.write(ROOT / f, arcname=str(Path("Final_Capstone_Project") / f))

print(f"Wrote {ZIP_NAME} with {len(INCLUDE)} files.")
with zipfile.ZipFile(ZIP_NAME) as zf:
    for n in zf.namelist():
        print("  ", n)
