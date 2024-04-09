import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms import CTransformers
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

def processing(file):
    text = ""
    for pdf in file:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    doc = text_splitter.split_text(text)
    embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectordb = FAISS.from_texts(doc, embedding)
    return vectordb

def llm1(retriever, query):
     llm1 = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.5, max_new_tokens=200
     )
    
     template = """
        Use the context to answer the question. Do not answer randomly if you don't know the answer.
        context:{context}
        question:{question}
        Helpful Answer:
        """
     prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
     chain_type_kwargs = {'prompt':prompt}

     qa_chain = RetrievalQA.from_chain_type(llm=llm1,
                                        chain_type='stuff',
                                        retriever=retriever.as_retriever(search_kwargs={'k':2}),
                                        return_source_documents=True,
                                        chain_type_kwargs=chain_type_kwargs)
     
     return qa_chain.invoke(query)

def llm_tamil(ans):
    llm = HuggingFaceEndpoint(
        repo_id="abhinand/gemma-2b-it-tamil-v0.1-alpha", temperature=0.5, max_new_tokens=200
    )

    # llm = CTransformers(model="tamil-llama-7b-instruct-v0.2.Q5_K_M.gguf",
    #                  model_type='llama')

    template1="""
    {ans} ஐ தமிழ் மொழியில் மொழிபெயர்க்கவும்
        """

    prompt1=PromptTemplate(template=template1,
                      input_variables=['ans'])

    return llm.invoke(prompt1.format(ans=ans))

def llm_telugu(ans):
    llm = HuggingFaceEndpoint(
        repo_id="abhinand/telugu-llama-7b-instruct-v0.1", temperature=0.5, max_new_tokens=200
    )
    template1="""
    {ans}ని తెలుగు భాషలోకి అనువదించండి
        """

    prompt1=PromptTemplate(template=template1,
                      input_variables=['ans'])

    return llm.invoke(prompt1.format(ans=ans))

def llm_malayalam(ans):
    llm = HuggingFaceEndpoint(
        repo_id="abhinand/malayalam-llama-7b-instruct-v0.1", temperature=0.5, max_new_tokens=200
    )
    template1="""
    {ans} മലയാളം ഭാഷയിലേക്ക് വിവർത്തനം ചെയ്യുക
        """

    prompt1=PromptTemplate(template=template1,
                      input_variables=['ans'])

    return llm.invoke(prompt1.format(ans=ans))

with st.sidebar:
    choice = st.radio('Languages', ['Tamil', 'Malayalam', 'Telugu'])
    # choice = st.radio('Languages', ['Tamil'])



def main():
    st.title('Q/A Vernacular Bot :books:')
    st.markdown("""
                    This bot will answers you in your selected language
                """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    file = st.file_uploader("Upload your PDFs here and click on 'Process'", type=["pdf"], accept_multiple_files=True)
    # if st.button("Process"):
            # with st.spinner("Processing"):
    if file is not None:
        with st.spinner("Processing"):
            processed_text = processing(file)
                 
    # if st.button('Process'):
        # processed_text = processing(file)
        st.markdown(""" 
                    :dart: Enter your query here :
                    """)
        query = st.text_input('What is your query ?')
        response = llm1(processed_text, query)
        ans = response['result']

        st.write(ans)

        if choice == 'Tamil':
            result = llm_tamil(ans)
            st.write(result)
            # st.session_state.conversation = llm_tamil(ans)

        elif choice == 'Malayalam':
            result = llm_malayalam(ans)
            st.write(result)
        #     st.session_state.conversation = llm_malayalam(ans)

        elif choice == 'Telugu':
            result = llm_telugu(ans)
            st.write(result)
        #     st.session_state.conversation = llm_telugu(ans)


if __name__ == '__main__':
    main()
