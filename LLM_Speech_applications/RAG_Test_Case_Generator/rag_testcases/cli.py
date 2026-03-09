
# import json

# from .config import validate_config, QUERIES
# from .models import load_vectorstore, load_reranker
# from .retrieval import build_bm25_corpus
# from .pipeline import run_and_validate


# def main():
#     validate_config()

#     print("✅ Loading vectorstore...")
#     vs = load_vectorstore()
#     bm25_model, bm25_docs = build_bm25_corpus(vs)
#     reranker = load_reranker()

#     print("✅ RAG test‑case generator (hybrid + rerank) ready.")

#     outputs = []
#     for q in QUERIES:
#         result = run_and_validate(vs, bm25_model, bm25_docs, reranker, q)
#         outputs.append(result)

#     # Save all query results into one file
#     with open("testcases_output.json", "w", encoding="utf-8") as f:
#         json.dump(outputs, f, indent=2, ensure_ascii=False)


import json
import os
from datetime import datetime

from .config import validate_config, QUERIES
from .models import load_vectorstore, load_reranker
from .retrieval import build_bm25_corpus
from .pipeline import run_and_validate


def main():
    validate_config()

    print("✅ Loading vectorstore...")
    vs = load_vectorstore()
    bm25_model, bm25_docs = build_bm25_corpus(vs)
    reranker = load_reranker()

    print("✅ RAG test‑case generator (hybrid + rerank) ready.")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    for i, q in enumerate(QUERIES, start=1):
        result = run_and_validate(vs, bm25_model, bm25_docs, reranker, q)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("outputs", f"testcases_q{i}_{ts}.json")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
