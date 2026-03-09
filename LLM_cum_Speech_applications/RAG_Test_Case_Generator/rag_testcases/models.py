
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from .config import EMBED_MODEL, FAISS_DIR, RERANK_MODEL_NAME, DEBUG_MODE


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_reranker() -> CrossEncoder:
    t0 = time.time()
    model = CrossEncoder(RERANK_MODEL_NAME)
    if DEBUG_MODE:
        print(f"✅ Loaded reranker '{RERANK_MODEL_NAME}' in {time.time() - t0:.2f}s")
    return model
