import cv2
import numpy as np
import matplotlib.pyplot as plt
# import cv2
import os
import json
import re
import csv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA


from paddleocr import PaddleOCR,draw_ocr
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')


from paddleocr import PaddleOCR,draw_ocr
ocr = PaddleOCR(use_angle_cls=True, lang='en')

embeddings=HuggingFaceBgeEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}

)

vectordb = FAISS.load_local("/mnt/newdisk/model/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever=vectordb.as_retriever()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.01,
    max_new_tokens=900
)


prompt_template = """
You are a knowledgeable assistant specialized in English language, medical data and text extraction and correction.
Extract the details in orderly manner and provide answer in the following format. Use context as reference and fetch correct medicine names from the vector database.
Question : {question}

context : {context}
Question : {question}

Date: Date mentioned in the prescription
Name of Hospital/Organization: Name of Hospital/Organization mentioned in the prescription
Country of Origin: Country of Origin mentioned in the prescription
Name of Doctor: Name of Doctor mentioned in the prescription
Doctor Registration Number: License or Registration number of the doctor mentioned in the prescription
Name of Patient: Name of the patient mentioned in the prescription
Name of Medicine: List of medicine names mentioned in the prescription
Quantity of Medicine: Quantity of Medicine mentioned in the prescription
Dosage of Medicine: Dosage of Medicine mentioned in the prescription

Helpful Answer :
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
chain_type_kwargs = {'prompt':prompt,"verbose": True}


retrievalQA = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k':1}),
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs)

# Function to preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = 255 - gray
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, gray = cv2.threshold(gray, 120, 255, cv2.THRESH_TOZERO)
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    kernel = np.ones((1, 50), np.uint8)
    eroded = cv2.erode(thresh_inv, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    preprocessed_image = cv2.subtract(gray_inv, dilated)
    return preprocessed_image

def remove_special_characters(sentence):
    pattern = r'[^a-zA-Z0-9\s]'
    cleaned_sentence = re.sub(pattern, '', sentence)

    return cleaned_sentence

def extract_data(data):

  query = "You are an expert in medical sciences and pharmacy. You are an expert in processing doctor's prescription. Process " + data + """ and give the output in the following pattern.

  For example, lets take Marrt. compare this name with the context. There is a medicine named Maviret. Marrt is actually a incomplete ocr translation by a ocr tool. So you have to compare the first 2 letters and last letter as reference for quicker comparison.
  Also, don't confuse between date and date of birth. Date means the date which the prescription has been prescribed. 
  The Registration number accompanies with No. before it generally. So if you see a number comes with No. before, then its mostly registration number.
  Unless its Phone No. which starts with Ph or M.
  The Organization name should be a noun. If random name without meaning is coming then its not a name.
  The medicine names are to be extracted from the rag architecture using retrieval mechanism. Only use the medicine names from the retrieval data.

  Use the details and update the following.
  Date:
  Name of Hospital/Organization:
  Country of Origin:
  Name of Doctor:
  Doctor Registration Number:
  Name of Patient:
  Name of Medicine:
  Quantity of Medicine:
  Dosage of Medicine:

  Only update the above required data. Do not provide random details like note, address of doctor, contact information, missing information etc.
  The output should follow the above format. Do not create duplicates or repeat yourself. Do not give answers in statements, only in above format.
  Do not repeat yourself. Include all medicine names in a single list.
  If medicine names are difficult to understand, try to compare the first 2 letters and last letter as reference for quicker comparison.

  """
  out1 = retrievalQA.invoke(query)
  out1 = out1['result']

  query = "You are an expert in medical sciences and pharmachy. You are an expert in processing doctor's prescription. Process" + out1 + " and give only the medicine names. No additional data are to be included"
  med_name = llm.invoke(query)

  query = "You are an expert in pharmachy and medicines. Use the medicine names, " + med_name + ", which are with spelling mistakes and compare them with the {context} and retrieve correct medicine names. Provide only the medicine name. Do not give any additional details."
  out3 = retrievalQA.invoke(query)
  med_result = out3['result']

  query = 'What is/are the correct medicine names mentioned in the following statement ' + med_result + ". Only give the correct medicine names. Do not mention statements like this is the correct medicine, just give the name etc."
  out4 = llm.invoke(query)
  print(out1)
  # print(out4)
  print(med_result)

  query = 'List of medicine names present in '+ out4 + ' in ascending order'
  names = llm.invoke(query)
  print(names)

  query = "You are an expert in English and prompt engineering. Process " + out1 + " and " + out4 + """ and update the following. Name all medicines in a single list.

  Use the details and update the following. Combine the data from both and update the following.

  Date: ,
  Name of Hospital/Organization:    ,
  Country of Origin:    ,
  Name of Doctor:   ,
  Doctor Registration Number:   ,
  Name of Patient:  ,
  Name of Medicine: ,
  Quantity of Medicine: ,
  Dosage of Medicine:

  Answer only in above format. Do not repeat yourself. The output should be precise. Don't add additional data like Note etc.
  """

  out5 = llm.invoke(query)

  return out5

# Function to extract text from images
def extract_text_from_images(image_path):
    preprocessed_image = preprocess_image(image_path)
    temp_image_path = os.path.join('/mnt/newdisk/model/', 'temp.jpg')
    cv2.imwrite(temp_image_path, preprocessed_image)

    # Extract text using PaddleOCR
    result = ocr.ocr(temp_image_path, cls=True)
    extracted_text = ''
    flattened_result = [item for sublist in result for item in sublist]
    for res in flattened_result:
        extracted_text += res[1][0] + '\n'

    data = "\n".join([line.strip() for line in extracted_text.splitlines()])

    output = extract_data(data)

    return output, data

def ans(image_path):
    final_output, raw = extract_text_from_images(image_path)
    return final_output, raw


