import nltk
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import PyPDF2

pdf_path = "CGS\llm\data\RachelGreenCV.pdf"

with open(pdf_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    pdf_text = ''

    for page_num in range(len(pdf_reader.pages)):
        pages = pdf_reader.pages[page_num]
        page_text = pages.extract_text()
        pdf_text += page_text

text = re.sub(r'^a-zA-Z', ' ', pdf_text)
text = re.sub(r'\[[0-9]*\]', ' ', text)
text = text.lower()
text = re.sub(r'/d+', ' ', text)
text = re.sub(r's+', ' ', text)

sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(word) for word in sentences if not word in stopwords.words('english')]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

similar = model.wv.most_similar('rachel')
print(similar)