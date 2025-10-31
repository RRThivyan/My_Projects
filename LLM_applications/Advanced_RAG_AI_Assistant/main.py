#!/usr/bin/env python3
# main.py — Unified FastAPI + CLI RAG app (full processing.py logic embedded)

import os
import sys
import json
import logging
import uuid
import pickle
from typing import Optional
from dotenv import load_dotenv

import numpy as np
import openai
from langdetect import detect
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rapidfuzz import fuzz

from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------
# Ensure working directory is script dir (fixes path issues under uvicorn)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ==================================
# Environment & Logging Setup
# ==================================
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

logging.basicConfig(
    filename=os.path.join(BASE_DIR, "rag_chatbot.log"),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

API_KEY = os.getenv("GPT4_API_KEY")
AZURE_ENDPOINT = os.getenv("GPT4_ENDPOINT")
DEPLOYMENT = os.getenv("GPT4_DEPLOYMENT", "gpt-4o")
API_VERSION = os.getenv("GPT4_API_VERSION", "2025-01-01-preview")

if not API_KEY or not AZURE_ENDPOINT:
    logging.error("API_KEY or AZURE_ENDPOINT is not set in .env file")
    sys.exit("Missing API key or Azure endpoint environment variables.")

# Models / files (absolute)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
FAISS_DIR = os.getenv("FAISS_DIR", os.path.join(BASE_DIR, "faiss_index_single_pdf"))
HISTORY_FILE = os.getenv("HISTORY_FILE", os.path.join(BASE_DIR, "chat_history.json"))
BM25_INDEX_FILE = os.getenv("BM25_INDEX_FILE", os.path.join(BASE_DIR, "bm25_index.pkl"))
DOCUMENTS_FILE = os.getenv("DOCUMENTS_FILE", os.path.join(BASE_DIR, "documents.pkl"))

# Expose variables for OpenAI Azure wrapper usage (processing uses openai.AzureOpenAI)
# We still keep openai package import for compatibility but processing uses AzureOpenAI class.
# (No extra action needed here other than env vars loaded.)

# ==================================
# Utility Functions (same as processing.py)
# ==================================
def expand_titles_bidirectionally(text):
    expansions = [text]
    lower = text.lower()
    if " and " in lower or " & " in lower:
        normalized = lower.replace(" & ", " and ")
        parts = [p.strip() for p in normalized.split(" and ") if p.strip()]
        for part in parts:
            try:
                idx = lower.find(part)
                prefix = text[:idx]
            except Exception:
                prefix = text
            candidate = (prefix + part).strip()
            if not candidate.lower().endswith("department"):
                expansions.append(candidate + " Department")
            expansions.append(candidate)
    seen = set()
    deduped = []
    for s in expansions:
        s_stripped = s.strip()
        if s_stripped.lower() not in seen:
            deduped.append(s_stripped)
            seen.add(s_stripped.lower())
    return deduped

def fuzzy_title_in_text(query, text, threshold=80):
    q = query.lower()
    t = text.lower()
    effective_threshold = threshold
    if "department" in q or "dept" in q or "director" in q:
        effective_threshold = min(80, threshold) - 5
    score = fuzz.token_sort_ratio(q, t)
    return score >= effective_threshold

# ==================================
# Load heavy models (embedding, faiss, reranker)
# This runs at import/startup so both CLI and FastAPI reuse same loaded objects
# ==================================
try:
    print("Loading embedding model:", EMBED_MODEL)
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print("Loading FAISS from:", FAISS_DIR)
    vector_db = FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)
    logging.info("✅ FAISS index loaded successfully.")
    print(f"✅ FAISS index loaded. docstore size: {len(vector_db.docstore._dict)}")
except Exception as e:
    logging.error(f"❌ Error loading FAISS index: {e}")
    sys.exit(f"Could not load vector database. Check FAISS path and permissions: {e}")

try:
    reranker = CrossEncoder(RERANK_MODEL)
    logging.info("✅ CrossEncoder reranker loaded successfully.")
    print("✅ CrossEncoder reranker loaded.")
except Exception as e:
    logging.error(f"❌ Error loading reranker: {e}")
    sys.exit(f"Could not load reranker model: {e}")

# ==================================
# BM25 Index Creation/Loading
# ==================================
def create_bm25_index():
    i = 0
    documents_local = []
    try:
        while True:
            documents_local.append(vector_db.docstore._dict[vector_db.index_to_docstore_id[i]])
            i += 1
    except (KeyError, IndexError):
        pass

    unique_docs = list({doc.page_content: doc for doc in documents_local}.values())
    augmented_texts = []
    for doc in unique_docs:
        text = doc.page_content.strip()
        augmented_texts.append(text)
        title_variants = expand_titles_bidirectionally(text)
        for v in title_variants:
            if v.strip() and v.strip() not in augmented_texts:
                augmented_texts.append(v.strip())
    tokenized_corpus = [t.split(" ") for t in augmented_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(bm25_index, f)
    with open(DOCUMENTS_FILE, "wb") as f:
        pickle.dump(unique_docs, f)
    return bm25_index, unique_docs

if not os.path.exists(BM25_INDEX_FILE) or not os.path.exists(DOCUMENTS_FILE):
    logging.info("Creating BM25 index (with title expansions)...")
    bm25, documents = create_bm25_index()
    logging.info("BM25 index created.")
else:
    with open(BM25_INDEX_FILE, "rb") as f:
        bm25 = pickle.load(f)
    with open(DOCUMENTS_FILE, "rb") as f:
        documents = pickle.load(f)

print(f"✅ BM25 loaded with {len(documents)} documents.")

# ==================================
# RAG Query Expansion & Search (same functions and prompts as processing.py)
# ==================================
def dynamic_expand_query_via_llm(query):
    prompt = (
        "You are an expert in Indian government office titles and official departments. "
        "Given the following query or role, generate a list of 5 likely synonyms, abbreviations, and alternate titles/phrases "
        "that could appear in government documents or records (one per line). Include abbreviations where common.\n\n"
        f"Original role/query: '{query}'\n\nList up to 5 alternatives, one per line."
    )
    try:
        client = openai.AzureOpenAI(
            api_key=API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=API_VERSION
        )
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120,
        )
        llm_expansions = response.choices[0].message.content.strip().split('\n')
        expansions = [query] + [v.strip() for v in llm_expansions if v.strip()]
        logging.info(f"Dynamic LLM expansions for query '{query}': {expansions}")
        return expansions
    except Exception as e:
        logging.error(f"Dynamic expansion error: {e}")
        return [query]

def dynamic_expand_query(query):
    base = dynamic_expand_query_via_llm(query)
    extra = expand_titles_bidirectionally(query)
    combined = list({q.strip() for q in (base + extra)})
    logging.info(f"Combined dynamic expansions for '{query}': {combined}")
    return combined

def expand_query(query):
    prompt = (
        "You are an expert in rewriting queries for information retrieval in government records and knowledge bases. "
        "Generate 3 diverse rewrites for the following user's query, considering alternate wording, possible synonyms, abbreviations, "
        "and equivalent government roles or titles.\n"
        f"Original query: '{query}'\nGenerate 3 variations separated by newlines."
    )
    try:
        client = openai.AzureOpenAI(
            api_key=API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=API_VERSION
        )
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=90,
        )
        llm_variations = [v.strip() for v in response.choices[0].message.content.strip().split('\n') if v.strip()]
    except Exception as e:
        logging.error(f"Query LLM rewrite error: {e}")
        llm_variations = []

    dynamic_synonyms = dynamic_expand_query(query)
    variations = list({query} | set(llm_variations) | set(dynamic_synonyms))
    logging.info(f"Full Query Expansion for '{query}': {variations}")
    return variations

def merge_semantic_roles(docs):
    merged = []
    seen_keys = set()
    for d in docs:
        txt = d.page_content.lower()
        if " and " in txt or " & " in txt:
            key = "compound_role"
        else:
            key = txt[:80].strip()
        if key not in seen_keys:
            merged.append(d)
            seen_keys.add(key)
    return merged

def rerank_context(query, docs, k=20):
    if not docs:
        return []
    try:
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)
        idx_sorted = np.argsort(scores)[::-1]
        return [docs[i] for i in idx_sorted[:k]]
    except Exception as e:
        logging.error(f"Rerank error: {e}")
        return docs[:k]

def search_rag(query, k=30, rerank=True):
    expansions = expand_query(query)
    vector_docs, bm25_docs = [], []
    for var in expansions:
        try:
            v = vector_db.similarity_search(var, k=k)
            vector_docs += v
        except Exception as e:
            logging.error(f"Vector similarity search error for '{var}': {e}")
        tokenized_var = var.split(" ")
        try:
            bm25_scores = bm25.get_scores(tokenized_var)
            top_n_indices = np.argsort(bm25_scores)[::-1][:k]
            bm25_docs += [documents[i] for i in top_n_indices]
        except Exception as e:
            logging.error(f"BM25 search error for '{var}': {e}")

    # debug: show counts & sample snippets (helps ensure same behavior)
    try:
        logging.info(f"search_rag: expansions={expansions}")
        logging.info(f"search_rag: vector_docs={len(vector_docs)}, bm25_docs={len(bm25_docs)}")
    except Exception:
        pass

    all_retrieved_docs = vector_docs + bm25_docs
    unique_docs = list({doc.page_content: doc for doc in all_retrieved_docs}.values())
    merged_docs = merge_semantic_roles(unique_docs)
    top_docs = rerank_context(query, merged_docs, k) if rerank else merged_docs[:k]
    return list(top_docs), expansions

def make_context_block(context_docs, all_queries, window_size=1):
    match_indices = set()
    doc_count = len(context_docs)
    for q in all_queries:
        for i, doc in enumerate(context_docs):
            if fuzzy_title_in_text(q, doc.page_content, threshold=80):
                match_indices.add(i)
    windowed = set(match_indices)
    for idx in match_indices:
        for delta in range(1, window_size + 1):
            if idx + delta < doc_count:
                windowed.add(idx + delta)
            if idx - delta >= 0:
                windowed.add(idx - delta)
    selected_docs = [context_docs[i] for i in sorted(windowed)] if match_indices else context_docs[:10]
    raw_texts = list({doc.page_content.strip() for doc in selected_docs})
    return "\n\n".join(raw_texts)

def detect_language(text):
    try:
        lang = detect(text)
        return "hi" if lang.startswith("hi") else "en"
    except Exception:
        return "en"

# ==================================
# LLM Chat Completion (preserve processing.py structured prompt)
# ==================================
def llm_chat_completion(history, user_message, context_block):
    query_lang = detect_language(user_message)

    # Structured system prompt with translation tags (kept identical to processing.py)
    system_prompt = {
      "system_prompt": {
        "identity": {
          "name": "Priya",
          "role": "Official Representative",
          "organization": "Government of Uttar Pradesh - Citizen Grievance and Information Support Portal"
        },
        "objective": (
          "Assist citizens on WhatsApp by providing clear, accurate, and verified responses "
          "regarding government departments, services, and grievance support. "
          "Responses must be strictly based on the provided RAG (retrieved) context or database information. "
          "If no relevant information is found, reply with: 'The information is not available in the provided records.' "
          "Do not speculate or generate content beyond the given data."
        ),
        "languages_supported": ["Hindi", "English"],
        "conversation_flow": {
          "start": "Incoming WhatsApp Message",
          "greeting": "[translate:नमस्ते! उत्तर प्रदेश सरकार के नागरिक शिकायत एवं सूचना सहायता पोर्टल में आपका स्वागत है। मैं प्रिया, आपकी सहायता के लिए उपस्थित हूँ। कृपया बताएं, मैं आपकी किस प्रकार मदद कर सकती हूँ?]",
          "intent_recognition": {
            "examples": [
              "Citizen grievance or complaint registration.",
              "Information about a specific government department.",
              "Status or contact details of an officer or department.",
              "Assistance with schemes, services, or application details.",
              "Request for escalation or further support."
            ]
          },
          "response_generation": {
            "instructions": [
              "Use the retrieved RAG chunks to formulate responses that are concise, contextually accurate, and factually verified.",
              "If the query mentions only part of a compound department name (e.g., 'Information Department' or 'Public Relations Department'), treat it as 'Information and Public Relations Department' only if the retrieved context supports it.",
              "Always prefix names with 'Shri' for males and 'Smt' for females.",
              "Respond in the same language as the user’s query. For example, if the user writes in Hindi, reply in Hindi; if in English, reply in English.",
              "End each conversation with a polite closing message."
            ]
          },
          "example_responses": {
            "available_information": [
              "[translate:श्री अमित कुमार, जनसंपर्क अधिकारी, सूचना एवं जनसंपर्क विभाग से संबद्ध हैं।]",
              "Smt. Neha Sharma is the designated officer for the Women Welfare Department in Lucknow district."
            ],
            "unavailable_information": [
              "The information is not available in the provided records.",
              "[translate:प्रदत्त अभिलेखों में यह जानकारी उपलब्ध नहीं है।]"
            ]
          },
          "end_of_chat": {
            "bot_prompt": "[translate:क्या मैं आपकी किसी और प्रकार से सहायता कर सकती हूँ?]",
            "closing_message_hindi": "[translate:उत्तर प्रदेश सरकार सहायता पोर्टल से संपर्क करने के लिए धन्यवाद। आपका दिन शुभ हो।]",
            "closing_message_english": "Thank you for contacting the Uttar Pradesh Government Support Portal. Have a good day."
          },
          "rules": [
            "Always reply in the same language as the query.",
            "Maintain a polite, respectful, and official tone throughout the interaction.",
            "Never speculate or infer information not present in the provided RAG context or database.",
            "If information is missing, respond exactly with: 'The information is not available in the provided records.'",
            "When a partial department name is mentioned, treat it as a compound title only when verified by context.",
            "Add 'Shri' before male names and 'Smt' before female names in all replies.",
            "Close all conversations with the official thank-you and goodbye message.",
            "Do not include lists, enumerations, or special formatting in user-facing responses.",
            "Avoid abbreviations, slang, or informal tone; maintain clarity and professionalism."
          ]
        }
      }
    }
    system_content_str = json.dumps(system_prompt, ensure_ascii=False)
    messages = [{"role": "system", "content": system_content_str}]
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in (history or [])])
    messages.append({
        "role": "user",
        "content": f"Retrieved Documents:\n{context_block}\n\nConversation History:\n{history_str}\n\nUser question: {user_message}"
    })
    try:
        client = openai.AzureOpenAI(
            api_key=API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=API_VERSION
        )
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        )
        reply = response.choices[0].message.content.strip()
        # Replace English closing with Hindi tag if user-language is Hindi and GPT returned English closing
        if query_lang == "hi" and "Thank you for contacting the Uttar Pradesh Government Support Portal. Have a good day." in reply:
            reply = reply.replace(
                "Thank you for contacting the Uttar Pradesh Government Support Portal. Have a good day.",
                "[translate:उत्तर प्रदेश सरकार सहायता पोर्टल से संपर्क करने के लिए धन्यवाद। आपका दिन शुभ हो।]"
            )
        return reply
    except Exception as e:
        logging.error(f"OpenAI chat completion error: {e}", exc_info=True)
        return "Sorry, the AI assistant is temporarily unavailable. Please try again later."

# ==================================
# Top-level process_user_query (same signature as processing.py)
# returns (history, ai_reply)
# ==================================
def process_user_query(user_query, history=None):
    history = history or []
    top_docs, expansions = search_rag(user_query, k=30, rerank=True)
    context_block = make_context_block(top_docs, expansions, window_size=1)
    ai_reply = llm_chat_completion(history, user_query, context_block)
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": ai_reply})
    logging.info(f"User: {user_query}\nAI: {ai_reply}")
    return history, ai_reply

# ==================================
# CLI helpers (keep same interactive behavior)
# ==================================
def print_chat_turn(user_query, ai_reply):
    print("\n" + "=" * 16)
    print("### 🧑 User Input")
    print(user_query)
    print("=" * 16)
    print("### 🤖 AI Response")
    print(ai_reply)
    print("=" * 16 + "\n")

def save_chat_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def rag_agentic_chatbot():
    chat_history = load_chat_history()
    print("Welcome to the Uttar Pradesh Government Support AI Assistant.")
    print("Type your question or 'exit' to quit.\n")
    while True:
        try:
            user_query = input("Your question: ").strip()
        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        if not user_query or user_query.lower() == 'exit':
            print("[translate:उत्तर प्रदेश सरकार सहायता पोर्टल से संपर्क करने के लिए धन्यवाद। आपका दिन शुभ हो।]")
            break
        chat_history, ai_reply = process_user_query(user_query, chat_history)
        save_chat_history(chat_history)
        print_chat_turn(user_query, ai_reply)

# ==================================
# FastAPI app (exposes /chat and /health)
# ==================================
app = FastAPI(title="Uttar Pradesh Government RAG AI Assistant API")
session_histories = {}

class ChatRequest(BaseModel):
    user_query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    user_query: str
    ai_reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    history = session_histories.get(session_id, [])
    updated_history, ai_reply = process_user_query(request.user_query, history)
    session_histories[session_id] = updated_history
    return ChatResponse(session_id=session_id, user_query=request.user_query, ai_reply=ai_reply)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "message": "✅ Uttar Pradesh Government RAG AI Assistant API is running.",
        "usage": "POST to /chat with {'user_query': 'your question'}"
    }

# ==================================
# Run: if executed directly, open CLI (keeps same behavior as processing.py).
# For API mode run via: uvicorn main:app --host 0.0.0.0 --port 8000
# ==================================
if __name__ == "__main__":
    # If user passes "--api" arg, start uvicorn programmatically (single-process, no reload)
    if len(sys.argv) > 1 and sys.argv[1] in ("--api", "api"):
        import uvicorn
        # ensure we run single-process to avoid deserialization issues
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
    else:
        try:
            rag_agentic_chatbot()
        except Exception as e:
            logging.error(f"Fatal error: {e}", exc_info=True)
            print("Fatal error: Please check logs for details.")
