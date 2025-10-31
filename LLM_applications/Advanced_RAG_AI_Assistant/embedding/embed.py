import os
import time
import re
import unicodedata
from datetime import timedelta
from tqdm import tqdm
import fitz  # PyMuPDF
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect, LangDetectException

PDF_FILE = "/mnt/data/thivyanfiles/website_rag/newfiles/final_data.pdf"
FAISS_DIR = "/mnt/data/thivyanfiles/website_rag/newfiles/faiss_index_single_pdf"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 32

os.makedirs(FAISS_DIR, exist_ok=True)
print(f"\n📘 Using embedding model: {MODEL_NAME}")
embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

def extract_text_pagewise(pdf_path: str):
    doc = fitz.open(pdf_path)
    page_blocks = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            page_blocks.append((i + 1, text))
    doc.close()
    return page_blocks

def normalize_text(text):
    return unicodedata.normalize('NFKC', text).replace('\u200c', '').strip()

def detect_language(text):
    text = text.strip()
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    return lang

def is_table_row_line(line):
    return line.strip().startswith("|") and line.strip().endswith("|") and "|" in line.strip()[1:-1]

def split_and_parse_contacts(page_num, lines):
    chunks = []
    # Detect and parse markdown pipe table
    for i, line in enumerate(lines):
        if is_table_row_line(line):
            # Parse table header and rows
            header_line = lines[i]
            if i+2 < len(lines) and is_table_row_line(lines[i+2]):
                header = [h.strip() for h in header_line.strip("|").split("|")]
                for j in range(i+2, len(lines)):
                    if is_table_row_line(lines[j]):
                        cells = [c.strip() for c in lines[j].strip("|").split("|")]
                        if len(cells) == len(header):
                            row_dict = dict(zip(header, cells))
                            row_dict["page"] = page_num
                            chunk_text = " | ".join(f"{k}: {v}" for k, v in row_dict.items())
                            chunks.append({
                                "text": chunk_text,
                                "type": "contact_row",
                                "page": page_num,
                                "language": detect_language(chunk_text),
                                "columns": row_dict
                            })
                    else:
                        break

    # Parse plain whitespace/tab-delimited contact/role lines (non-table)
    contact_pattern = re.compile(
        r"(?:^\d+\.\s+|^)(Shri|Smt|Dr\.?)?\s*([\w\s\.\'\-]+)\s+(Chief Minister|Deputy Chief Minister|Cabinet Minister|Minister|Secretary|OSD|Joint Secretary|Under Secretary)?\s*([0-9, \-\(\)\/\.]+)?\s*(.*)$",
        re.UNICODE
    )

    for line in lines:
        if not is_table_row_line(line) and contact_pattern.match(line.strip()):
            parts = contact_pattern.match(line.strip())
            if parts:
                name = parts.group(2).strip() if parts.group(2) else ""
                designation = parts.group(3).strip() if parts.group(3) else ""
                phone = parts.group(4).strip() if parts.group(4) else ""
                other = parts.group(5).strip() if parts.group(5) else ""
                chunk_text = f"Name: {name} | Designation: {designation} | Phone: {phone} | Other: {other} | page: {page_num}"
                row_dict = {
                    "Name": name,
                    "Designation": designation,
                    "Phone": phone,
                    "Other": other,
                    "page": page_num
                }
                chunks.append({
                    "text": chunk_text,
                    "type": "contact_whitespace",
                    "page": page_num,
                    "language": detect_language(chunk_text),
                    "columns": row_dict
                })
    return chunks

def chunk_text_table_and_contact_aware(page_num, text: str, chunk_size: int = 350, overlap: int = 40):
    lines = text.split('\n')
    chunks = []
    contacts = split_and_parse_contacts(page_num, lines)
    chunks.extend(contacts)
    # Fallback: Also add general narrative semantic chunks
    for block in RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", "।", "!", "?"]).split_text(text):
        norm = normalize_text(block)
        if norm and len(norm) > 50:  # Avoid storing duplicates for short contact blocks
            lang = detect_language(norm)
            chunks.append({
                "text": norm,
                "type": "text",
                "header": None,
                "page": page_num,
                "language": lang
            })
    return chunks

def create_documents_from_pdf(pdf_path: str):
    print(f"📄 Extracting and splitting text from {os.path.basename(pdf_path)} ...")
    full_chunks = []
    page_blocks = extract_text_pagewise(pdf_path)
    for page_num, text in page_blocks:
        page_chunks = chunk_text_table_and_contact_aware(page_num, text)
        for chunk in page_chunks:
            metadata = {
                "source": os.path.basename(pdf_path),
                "type": chunk["type"],
                "language": chunk.get("language"),
                "page": chunk.get("page")
            }
            # Add all column fields for contact rows
            if "columns" in chunk:
                metadata.update(chunk["columns"])
            full_chunks.append(Document(page_content=chunk["text"], metadata=metadata))
    print(f"✅ Created {len(full_chunks)} enriched chunks.")
    return full_chunks

def embed_in_batches(docs, model, batch_size=32):
    all_embeddings = []
    texts = [doc.page_content for doc in docs]
    for i in tqdm(range(0, len(texts), batch_size), desc="🔢 Embedding batches"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
    return all_embeddings

if __name__ == "__main__":
    print("\n🚀 Starting atomic contact/narrative-aware embedding pipeline...\n")
    start_time = time.time()
    docs = create_documents_from_pdf(PDF_FILE)
    print("\n⚙️ Generating embeddings ...")
    embed_start = time.time()
    embeddings = embed_in_batches(docs, embedding_model, batch_size=BATCH_SIZE)
    embed_end = time.time()
    print(f"✅ Embeddings finished in {timedelta(seconds=int(embed_end - embed_start))}.")
    print("\n📦 Building FAISS vector index ...")
    text_embedding_pairs = list(zip([doc.page_content for doc in docs], embeddings))
    metadata_list = [doc.metadata for doc in docs]
    vector_db = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embedding_model,
        metadatas=metadata_list
    )
    vector_db.save_local(FAISS_DIR)
    total_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\n🎯 FAISS index saved at: {FAISS_DIR}")
    print(f"🕒 Total processing time: {total_time}")
    print("\n✅ Contact-row, narrative, atomic embedding pipeline complete.")
