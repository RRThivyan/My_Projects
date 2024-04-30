import pandas as pd

messages = pd.read_csv('NLP\spam.csv',
                        encoding='latin1',
                       names=['label', 'message'])


# Data Cleaning and preprocessing
import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
lem = WordNetLemmatizer()
corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()

    # review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = [lem.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ''.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)
x = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,:1].values

print('Length of x: ', len(x))
print('Length of y : ', len(y))

from sklearn.model_selection import train_test_split as tts

xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.2, random_state=1)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(xtrain, ytrain)

ypred = spam_detect_model.predict(xtest)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest,ypred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(ytest, ypred)
print(accuracy)