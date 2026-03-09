# # app.py
# import json
# from datetime import datetime

# import streamlit as st

# from rag_testcases.config import (
#     QUERIES,
#     validate_config,
#     DEBUG_MODE,
# )
# from rag_testcases.models import load_vectorstore, load_reranker
# from rag_testcases.retrieval import build_bm25_corpus
# from rag_testcases.pipeline import run_single_query


# @st.cache_resource(show_spinner=True)
# def init_pipeline():
#     """Load config, vectorstore, BM25, and reranker once per process."""
#     validate_config()
#     vs = load_vectorstore()
#     bm25_model, bm25_docs = build_bm25_corpus(vs)
#     reranker = load_reranker()
#     return vs, bm25_model, bm25_docs, reranker


# def main():
#     st.set_page_config(
#         page_title="RAG Test Case Generator",
#         layout="wide",
#     )

#     st.title("RAG Test Case Generator")
#     st.markdown(
#         "Generate **use cases & test cases** from your indexed PRD/screenshots "
#         "using a hybrid RAG pipeline."
#     )

#     with st.sidebar:
#         st.header("Run settings")

#         use_default_queries = st.checkbox(
#             "Use predefined queries",
#             value=True,
#             help="If unchecked, you can type a custom query.",
#         )

#         if use_default_queries:
#             query = st.selectbox(
#                 "Select query",
#                 options=QUERIES,
#                 index=0,
#             )
#         else:
#             query = st.text_area(
#                 "Custom query",
#                 value="Generate positive, negative, and boundary test cases for ...",
#                 height=140,
#             )

#         run_button = st.button("Run generator", type="primary")

#     if not run_button:
#         st.info("Select or enter a query, then click **Run generator**.")
#         return

#     if not query or not query.strip():
#         st.error("Query cannot be empty.")
#         return

#     with st.spinner("Loading vectorstore and reranker..."):
#         vs, bm25_model, bm25_docs, reranker = init_pipeline()

#     st.write("### Query")
#     st.code(query)

#     with st.spinner("Running RAG pipeline (retrieve → rerank → generate JSON)..."):
#         result = run_single_query(vs, bm25_model, bm25_docs, reranker, query)

#     st.write("### Generated JSON")
#     st.json(result)

#     # Prepare downloadable JSON
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"testcases_{ts}.json"
#     file_bytes = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")

#     st.download_button(
#         label=f"Download JSON ({filename})",
#         data=file_bytes,
#         file_name=filename,
#         mime="application/json",
#     )

#     if DEBUG_MODE:
#         st.caption("DEBUG_MODE is ON; see server logs for detailed retrieval and generation traces.")


# if __name__ == "__main__":
#     main()


# app.py
import json
from datetime import datetime

import streamlit as st

from rag_testcases.config import (
    QUERIES,
    validate_config,
    DEBUG_MODE,
)
from rag_testcases.models import load_vectorstore, load_reranker
from rag_testcases.retrieval import build_bm25_corpus
from rag_testcases.pipeline import run_single_query


@st.cache_resource(show_spinner=True)
def init_pipeline():
    """Load config, vectorstore, BM25, and reranker once per process."""
    validate_config()
    vs = load_vectorstore()
    bm25_model, bm25_docs = build_bm25_corpus(vs)
    reranker = load_reranker()
    return vs, bm25_model, bm25_docs, reranker


def main():
    st.set_page_config(
        page_title="RAG Test Case Generator",
        layout="wide",
    )

    st.title("RAG Test Case Generator")
    st.markdown(
        "Generate **use cases & test cases** from your indexed PRD/screenshots "
        "using a hybrid RAG pipeline."
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Run settings")

        mode = st.radio(
            "Query source",
            options=["Predefined", "Custom"],
            index=0,
            help="Use one of the built‑in demo queries or type your own.",
        )

        if mode == "Predefined":
            query = st.selectbox(
                "Select predefined query",
                options=QUERIES,
                index=0,
            )
        else:
            query = st.text_area(
                "Custom query",
                value="Generate positive, negative, and boundary test cases for ...",
                height=140,
            )

        run_button = st.button("Run generator", type="primary")

    # Guard: wait for click
    if not run_button:
        st.info("Select or enter a query, then click **Run generator**.")
        return

    # Guard: non-empty query
    if not query or not query.strip():
        st.error("Query cannot be empty.")
        return

    # Init pipeline
    with st.spinner("Loading vectorstore and reranker..."):
        vs, bm25_model, bm25_docs, reranker = init_pipeline()

    st.write("### Query")
    st.code(query)

    # Run pipeline
    with st.spinner("Running RAG pipeline (retrieve → rerank → generate JSON)..."):
        result = run_single_query(vs, bm25_model, bm25_docs, reranker, query)

    # Show result
    st.write("### Generated JSON")
    st.json(result)

    # Prepare downloadable JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"testcases_{ts}.json"
    file_bytes = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")

    st.download_button(
        label=f"Download JSON ({filename})",
        data=file_bytes,
        file_name=filename,
        mime="application/json",
    )

    if DEBUG_MODE:
        st.caption(
            "DEBUG_MODE is ON; see server logs for detailed retrieval and generation traces."
        )


if __name__ == "__main__":
    main()
