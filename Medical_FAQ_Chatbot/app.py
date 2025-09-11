import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Medical FAQ Chatbot",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- CUSTOM STYLING ----------------------
st.markdown(
    """
    <style>
    .title {
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #2C3E50;
        text-align: center;
    }
    .subtitle {
        font-size: 18px !important;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">💊 RAG-based Medical FAQ Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Get quick, factual answers to your medical questions</p>', unsafe_allow_html=True)

# ---------------------- LOAD ENV & MODELS ----------------------
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

endpoint = os.getenv("OPENAI_API_ENDPOINT")
api_key = os.getenv("OPENAI_API_KEY")

llm = None
if endpoint and api_key:
    try:
        azure_endpoint_base = "/".join(endpoint.split("/")[:3])
        deployment_name = endpoint.split("/")[5]
        api_version = endpoint.split("=")[-1]

        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint_base,
            api_key=api_key,
            azure_deployment=deployment_name,
            api_version=api_version,
            temperature=0.7,
        )
    except IndexError:
        st.error("❌ Could not parse Azure OpenAI endpoint URL. Check your .env file.")
else:
    st.error("❌ Azure OpenAI credentials not found in .env file.")

try:
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    st.error(f"❌ Failed to load vector store: {e}")
    db = None

# ---------------------- MEMORY & CHAIN ----------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", return_messages=True
    )

retriever = None
if db:
    retriever = db.as_retriever(search_kwargs={"k": 2})

chain = None
if llm and retriever:
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        verbose=True,
    )

# ---------------------- HELPER FUNCTION ----------------------
def get_response_from_query(chain, query):
    if chain is None:
        return "⚠️ Chain not initialized. Please check your configuration.", []

    result = chain({"question": query})
    response = result["answer"]
    
    # Improved "I don't know" handling
    if "don't know" in response.lower() or "no information" in response.lower():
        response = "I couldn't find a specific answer in my knowledge base. Could you please rephrase your question or ask about a different topic?"
        
    return response, result.get("source_documents", [])

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
    st.markdown("### ⚕️ About")
    st.write(
        "This chatbot uses **RAG (Retrieval-Augmented Generation)** with Azure OpenAI "
        "and FAISS to answer medical FAQs."
    )
    st.markdown("### ⚙️ Settings")
    if retriever:
        retriever.search_kwargs["k"] = st.slider("Number of documents to search:", 1, 5, 2)

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()

# ---------------------- CHAT INTERFACE ----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.chat_input("Ask a medical question...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("🤔 Thinking..."):
        response, docs = get_response_from_query(chain, question)

    st.session_state.chat_history.append({"role": "assistant", "content": response, "sources": docs})

# ---------------------- DISPLAY CHAT ----------------------
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        if chat["role"] == "assistant":
            st.markdown(
                f"""
                <div style="background-color:#ECF0F1;padding:15px;border-radius:10px;margin:10px 0;">
                    {chat['content']}
                </div>
                """,
                unsafe_allow_html=True
            )
            if "sources" in chat and chat["sources"]:
                with st.expander("📚 Sources"):
                    for i, d in enumerate(chat["sources"], 1):
                        st.markdown(f"**Source {i}:** {d.page_content[:300]}...")
        else:
            st.markdown(chat["content"])
