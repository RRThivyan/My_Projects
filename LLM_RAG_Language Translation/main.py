import os
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY')

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader('data').load_data()

index = VectorStoreIndex.from_documents(documents, show_progress=True)

query_engine = index.as_query_engine()

response = query_engine.query('What are the job postings for vacancy')

print(response)

print(50*'-')

# from llama_index.core.response.pprint_utils import pprint_response

# pprint_response(response, show_source=True)

# from llama_index.core.retrievers import VectorIndexAutoRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.indices.postprocessor import SimilarityPostprocessor

# retriever = VectorIndexAutoRetriever(index=index, similarity_top_k=4)

# query_engine = RetrieverQueryEngine(retriever=retriever)

# response = query_engine.query('What are the job postings for vacancy')

# print(response)
