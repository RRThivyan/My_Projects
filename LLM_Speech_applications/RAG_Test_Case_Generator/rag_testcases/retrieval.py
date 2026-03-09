
import time
from typing import List, Any
from rank_bm25 import BM25Okapi

from .config import TOP_K_VECTOR, TOP_K_KEYWORD, TOP_K_RERANKED, DEBUG_MODE


def build_bm25_corpus(vectorstore):
    """Prepare BM25 over all docstore documents."""
    docs = list(vectorstore.docstore._dict.values())
    tokenized = [d.page_content.split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, docs


def hybrid_retrieve(vectorstore, bm25_model, bm25_docs, query: str):
    """Hybrid retrieval: vector + BM25 keyword, merged and deduplicated (pre-rerank)."""
    retriever_vec = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_VECTOR},
    )

    t0 = time.time()
    docs_vec = retriever_vec.invoke(query)
    vec_latency = time.time() - t0

    t1 = time.time()
    scores = bm25_model.get_scores(query.split())
    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:TOP_K_KEYWORD]
    docs_kw = [bm25_docs[i] for i in top_indices]
    kw_latency = time.time() - t1

    merged = []
    seen = set()
    for d in docs_vec + docs_kw:
        key = d.page_content.strip()
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)

    if DEBUG_MODE:
        print(
            f"\n[DEBUG] Vector docs={len(docs_vec)} ({vec_latency:.3f}s), "
            f"Keyword docs={len(docs_kw)} ({kw_latency:.3f}s), "
            f"Merged(before rerank)={len(merged)}"
        )

    return merged


def rerank_docs(reranker, query: str, docs) -> List[Any]:
    """Rerank merged docs using a cross-encoder and keep top-k."""
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    t0 = time.time()
    scores = reranker.predict(pairs)
    rerank_latency = time.time() - t0

    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in scored[:TOP_K_RERANKED]]

    if DEBUG_MODE:
        print(
            f"[DEBUG] Reranker scored {len(docs)} docs in {rerank_latency:.3f}s; "
            f"kept top {len(top_docs)}"
        )

    return top_docs
