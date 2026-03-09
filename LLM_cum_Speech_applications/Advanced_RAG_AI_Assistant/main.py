import os  
import sys  
import json  
import logging  
from typing import List, Optional, Dict  
  
from fastapi import FastAPI  
from fastapi.responses import JSONResponse  
from pydantic import BaseModel  
  
from dotenv import load_dotenv  
import numpy as np  
import openai  
from langdetect import detect  
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  
from sentence_transformers import CrossEncoder  
from rank_bm25 import BM25Okapi  
import pickle  
from rapidfuzz import fuzz  
  
# ================================  
# Environment & Logging Setup  
# ================================  
load_dotenv()  
logging.basicConfig(  
    filename='rag_chatbot.log',  
    level=logging.INFO,  
    format='%(asctime)s [%(levelname)s] %(message)s'  
)  
  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
def abs_path(filename):  
    return os.path.join(BASE_DIR, filename)  
  
API_KEY = os.getenv("GPT4_API_KEY")  
AZURE_ENDPOINT = os.getenv("GPT4_ENDPOINT")  
DEPLOYMENT = "gpt-4o"  
API_VERSION = "2025-01-01-preview"  
  
if not API_KEY or not AZURE_ENDPOINT:  
    logging.error("API_KEY or AZURE_ENDPOINT is not set in .env file")  
    sys.exit("Missing API key or Azure endpoint environment variables.")  
  
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  
FAISS_DIR = abs_path("faiss_index_single_pdf")  
HISTORY_FILE = abs_path("chat_history.json")  
BM25_INDEX_FILE = abs_path("bm25_index.pkl")  
DOCUMENTS_FILE = abs_path("documents.pkl")  
  
# ================================  
# Persistent History Functions  
# ================================  
def save_chat_history(history, HISTORY_FILE):  
    try:  
        with open(HISTORY_FILE, "w") as f:  
            json.dump(history, f)  
    except Exception as e:  
        logging.error(f"Error saving chat history: {e}")  
  
def load_chat_history(HISTORY_FILE):  
    try:  
        if os.path.exists(HISTORY_FILE):  
            with open(HISTORY_FILE, "r") as f:  
                return json.load(f)  
    except Exception as e:  
        logging.error(f"Error loading chat history: {e}")  
    return []  
  
# Global persistent history  
persistent_history = load_chat_history(HISTORY_FILE)  
  
# ================================  
# Utility Functions  
# ================================  
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
    keywords = ["chief minister", "cm", "head of state", "hon'ble chief minister", "yogi adityanath"]  
    for word in keywords:  
        if word in q and word in t:  
            return True  
    effective_threshold = threshold  
    if "department" in q or "dept" in q or "director" in q:  
        effective_threshold = min(80, threshold) - 5  
    score = fuzz.token_sort_ratio(q, t)  
    return score >= effective_threshold  
  
# ================================  
# Load Embedding Model & FAISS Index  
# ================================  
try:  
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)  
    vector_db = FAISS.load_local(  
        FAISS_DIR, embedding_model, allow_dangerous_deserialization=True  
    )  
    logging.info("✅ FAISS index loaded successfully.")  
except Exception as e:  
    logging.error(f"❌ Error loading FAISS index: {e}")  
    raise RuntimeError("Could not load vector database. Check FAISS path and permissions.")  
  
try:  
    reranker = CrossEncoder(RERANK_MODEL)  
    logging.info("✅ CrossEncoder reranker loaded successfully.")  
except Exception as e:  
    logging.error(f"❌ Error loading reranker: {e}")  
    raise RuntimeError("Could not load reranker model.")  
  
# ================================  
# BM25 Index Creation/Loading  
# ================================  
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
  
# ================================  
# RAG Query Expansion & Search  
# ================================  
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
        return expansions  
    except Exception as e:  
        logging.error(f"Dynamic expansion error: {e}")  
        return [query]  
  
def dynamic_expand_query(query):  
    base = dynamic_expand_query_via_llm(query)  
    extra = expand_titles_bidirectionally(query)  
    combined = list({q.strip() for q in (base + extra)})  
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
            vector_docs += vector_db.similarity_search(var, k=k)  
        except Exception as e:  
            logging.error(f"Vector similarity search error for '{var}': {e}")  
        tokenized_var = var.split(" ")  
        try:  
            bm25_scores = bm25.get_scores(tokenized_var)  
            top_n_indices = np.argsort(bm25_scores)[::-1][:k]  
            bm25_docs += [documents[i] for i in top_n_indices]  
        except Exception as e:  
            logging.error(f"BM25 search error for '{var}': {e}")  
  
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
  
# ================================  
# Build Messages for LLM (History as List)  
# ================================  
def build_llm_messages(system_prompt_str, history, user_query, context_block):  
    messages = [{"role": "system", "content": system_prompt_str}]  
    if context_block.strip():  
        messages.append({"role": "assistant", "content": f"Retrieved Documents:\n{context_block}"})  
    for turn in history:  
        messages.append({"role": turn["role"], "content": turn["content"]})  
    messages.append({"role": "user", "content": user_query})  
    return messages  
  
def llm_chat_completion(history, user_message, context_block):  
    query_lang = detect_language(user_message)  
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
    messages = build_llm_messages(system_content_str, history, user_message, context_block)  
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
        if query_lang == "hi" and "Thank you for contacting the Uttar Pradesh Government Support Portal. Have a good day." in reply:  
            reply = reply.replace(  
                "Thank you for contacting the Uttar Pradesh Government Support Portal. Have a good day.",  
                "[translate:उत्तर प्रदेश सरकार सहायता पोर्टल से संपर्क करने के लिए धन्यवाद। आपका दिन शुभ हो।]"  
            )  
        return reply  
    except Exception as e:  
        logging.error(f"OpenAI chat completion error: {e}")  
        return "Sorry, the AI assistant is temporarily unavailable. Please try again later."  
  
def process_user_query(user_query, history=None):  
    history = history or []  
    top_docs, expansions = search_rag(user_query, k=30, rerank=True)  
    context_block = make_context_block(top_docs, expansions, window_size=1)  
    ai_reply = llm_chat_completion(history, user_query, context_block)  
    history.append({"role": "user", "content": user_query})  
    history.append({"role": "assistant", "content": ai_reply})  
    logging.info(f"User: {user_query}\nAI: {ai_reply}")  
    return history, ai_reply  
  
# ================================  
# FastAPI Endpoint Definition  
# ================================  
app = FastAPI(  
    title="Uttar Pradesh Gov RAG Chatbot API",  
    description="API endpoint for RAG-based WhatsApp Government Chatbot",  
    version="1.0"  
)  
  
class ChatRequest(BaseModel):  
    user_query: str  
    history: Optional[List[Dict[str, str]]] = None  
  
@app.post("/chat")  
async def chat_endpoint(request: ChatRequest):  
    global persistent_history  
    history = request.history if request.history else persistent_history  
    try:  
        history, ai_reply = process_user_query(request.user_query, history)  
        persistent_history = history  
        save_chat_history(persistent_history, HISTORY_FILE)  
        return {"reply": ai_reply}  
    except Exception as e:  
        logging.error(f"API error: {e}")  
        return JSONResponse(status_code=500, content={  
            "reply": "Sorry, the AI assistant is temporarily unavailable. Please try again later."  
        })  
  
@app.get("/")  
async def root():  
    return {"status": "UP Gov RAG Chatbot API is running."}  
  
@app.get("/health")  
async def health():  
    return {"status": "ok"}  
  
if __name__ == "__main__":  
    import uvicorn  
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)  