
from typing import Dict, Any

from .config import MIN_EVIDENCE, DEBUG_MODE, MAX_USE_CASES
from .prompting import format_docs
from .retrieval import hybrid_retrieve, rerank_docs
from .llm import call_llm
from .evaluation import validate_output_schema


def _limit_use_cases(result: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp number of use_cases to MAX_USE_CASES from config."""
    use_cases = result.get("use_cases")
    if isinstance(use_cases, list) and len(use_cases) > MAX_USE_CASES:
        result["use_cases"] = use_cases[:MAX_USE_CASES]
    return result


def run_single_query(vectorstore, bm25_model, bm25_docs, reranker, query: str) -> Dict[str, Any]:
    """Hybrid retrieve → rerank → build context → LLM → JSON."""
    merged_docs = hybrid_retrieve(vectorstore, bm25_model, bm25_docs, query)
    reranked_docs = rerank_docs(reranker, query, merged_docs)

    if len(reranked_docs) < MIN_EVIDENCE:
        if DEBUG_MODE:
            print("[DEBUG] Low evidence after rerank; using LOW_EVIDENCE context")
        context_str = "LOW_EVIDENCE"
    else:
        context_str = format_docs(reranked_docs)
        if DEBUG_MODE:
            print(
                f"[DEBUG] Final context uses {len(reranked_docs)} chunks; "
                f"context length={len(context_str)} chars"
            )

    result = call_llm(context_str, query)
    result = _limit_use_cases(result)
    return result


def run_and_validate(vectorstore, bm25_model, bm25_docs, reranker, query: str) -> Dict[str, Any]:
    print(f"\n🔍 Query:\n{query}\n")
    result = run_single_query(vectorstore, bm25_model, bm25_docs, reranker, query)
    errors = validate_output_schema(result)

    if errors:
        print("⚠️ Schema/Eval issues:")
        for e in errors:
            print(" -", e)
    else:
        print("✅ Basic JSON schema checks passed.")

    return result
