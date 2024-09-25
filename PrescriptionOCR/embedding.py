from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

doc_path = input('Enter the document folder directory : ')

# doc_path = r'/mnt/newdisk/model/doc'

# Read the pdfs from the folder
loader=PyPDFDirectoryLoader(doc_path)

documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

final_documents=text_splitter.split_documents(documents)

## Embedding Using Huggingface
embeddings=HuggingFaceBgeEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

vectordb = FAISS.from_documents(final_documents, embeddings)

vectordb.save_local("/mnt/newdisk/model/faiss_index")
