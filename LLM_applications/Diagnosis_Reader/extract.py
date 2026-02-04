# ============================================================
# Medical PDF → Doctor Summary (Azure OpenAI, Config via .env)
# ============================================================

from __future__ import annotations

import os
import re
import requests
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv
from datetime import datetime
import fitz  # PyMuPDF

# ============================================================
# LOAD CONFIG FROM .env
# ============================================================

load_dotenv()  # Load environment variables from .env

# ---------- Timestamp for output files ----------
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------- File paths ----------
PDF_PATH = Path(os.getenv("PDF_PATH", "Debadrita_Female_12_2024_10.pdf")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SUMMARY_PATH = OUTPUT_DIR / f"{PDF_PATH.stem}_doctor_summary_{TIMESTAMP}.txt"
OUTPUT_EXTRACTED_TEXT = OUTPUT_DIR / f"{PDF_PATH.stem}_extracted_{TIMESTAMP}.txt"

# ---------- Azure OpenAI ----------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
API_VERSION = os.getenv("API_VERSION")
API_KEY = os.getenv("API_KEY")

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 3000))

# ---------- Safety limits ----------
MAX_EXTRACTED_CHARS = 200_000
MAX_LINE_LEN = 600

# ============================================================
# STEP 1: FAST PDF ANALYSIS (NO OCR)
# ============================================================

def analyze_pdf_text(pdf_path: Path) -> Tuple[str, bool]:
    """
    Returns:
        extracted_text, is_digital
    """
    doc = fitz.open(pdf_path)
    text_pages = 0
    text_chunks = []

    for page in doc:
        txt = page.get_text("text").strip()
        if len(txt) > 50:
            text_pages += 1
            text_chunks.append(txt)

    full_text = "\n\n".join(text_chunks)
    is_digital = text_pages >= 1

    print(
        f"PDF analysis → pages={len(doc)}, "
        f"text_pages={text_pages}, chars={len(full_text)}"
    )

    return full_text, is_digital

# ============================================================
# STEP 2: OPTIONAL OCR WITH DOCLING (SAFE MODE)
# ============================================================

def extract_with_docling(pdf_path: Path) -> str:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

    pdf_opts = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=False,  # 🚑 avoids crash
    )
    pdf_opts.ocr_options.force_full_page_ocr = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_opts,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

    result = converter.convert(pdf_path)
    return result.document.export_to_markdown() or ""

# ============================================================
# STEP 3: CLEAN & NORMALIZE TEXT
# ============================================================

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\bpage\s*\d+(\s*of\s*\d+)?\b", " ", text, flags=re.I)
    text = re.sub(r"[\-_]{8,}", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > MAX_LINE_LEN:
            line = line[:MAX_LINE_LEN]
        lines.append(line)

    cleaned = "\n".join(lines)
    return cleaned[:MAX_EXTRACTED_CHARS]

# ============================================================
# STEP 4: BUILD DOCTOR PROMPT (NO INTERPRETATION)
# ============================================================

def build_doctor_prompt(extracted_text: str) -> str:
    return f"""
You are preparing a concise, doctor-facing summary from a diagnostic laboratory report.

STRICT RULES:
- Use ONLY the information present in the report text.
- Do NOT infer diagnoses.
- Do NOT recommend medications or treatments.
- Do NOT explain medical meaning.
- Preserve exact values, units, ranges, and flags.
- Output plain text only.

Write using the following sections EXACTLY ONCE and IN THIS ORDER:

PATIENT & REPORT DETAILS
EXTRACTED TEST RESULTS

Under EXTRACTED TEST RESULTS:
- Group tests logically (e.g., CRP, Hemogram, Urine).
- Use tables where the report clearly contains tables.
- Do NOT mark anything as abnormal unless the report explicitly does.

REPORT TEXT:
{extracted_text}
""".strip()

# ============================================================
# STEP 5: CALL AZURE OPENAI
# ============================================================

def call_azure_openai(prompt: str) -> str:
    url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
        f"{DEPLOYMENT_NAME}/chat/completions"
        f"?api-version={API_VERSION}"
    )

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a careful medical summarization assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"Azure OpenAI error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("Starting PDF extraction...")

    text, is_digital = analyze_pdf_text(PDF_PATH)

    if is_digital:
        print("Digital text detected → OCR disabled")
        extracted = text
    else:
        print("No usable digital text → running OCR")
        extracted = extract_with_docling(PDF_PATH)

    extracted = clean_text(extracted)
    OUTPUT_EXTRACTED_TEXT.write_text(extracted, encoding="utf-8")

    print("Calling Azure OpenAI...")
    prompt = build_doctor_prompt(extracted)
    summary = call_azure_openai(prompt)
    OUTPUT_SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print("\n===== DOCTOR SUMMARY =====\n")
    print(summary)
    print(f"\nSaved to: {OUTPUT_SUMMARY_PATH}")

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
