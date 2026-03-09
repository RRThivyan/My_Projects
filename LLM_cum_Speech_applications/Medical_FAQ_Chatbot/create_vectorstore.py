import os
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Set your OpenAI API key
# The key is loaded automatically by the OpenAI library from the .env file

def create_vector_store():
    """
    Reads medical FAQ data from a CSV, creates embeddings, and saves them to a FAISS vector store.
    """
    # Load the dataset
    try:
        df = pd.read_csv('medicalqa.csv')
        print("CSV loaded successfully.")
    except FileNotFoundError:
        print("Error: medicalqa.csv not found. Make sure it's in the same directory as the script.")
        return

    # Combine question and answer for context
    df['text'] = df['Question'] + " " + df['Answer']

    # Load documents from the DataFrame
    loader = DataFrameLoader(df, page_content_column='text')
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    # Create Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name='all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Initialized Hugging Face embeddings.")

    # Create FAISS vector store
    db = FAISS.from_documents(docs, embeddings)
    print("FAISS vector store created.")

    # Save the vector store
    db.save_local("faiss_index")
    print("Vector store saved to 'faiss_index' directory.")

if __name__ == "__main__":
    create_vector_store()
