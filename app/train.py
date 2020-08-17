import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn import metrics
import nltk
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
import re
from sklearn.neighbors import *
from sklearn.tree import *

stop_words = nltk.corpus.stopwords.words('english')
porter = nltk.PorterStemmer()
    

def trainmodel():
    #1.Text Processing
    df = pd.read_table('app/SMSSpamCollection.txt', header=None)
    y = df[0]
    #print(df.head())
    #df.info()
    #Label encoders -converting Spam as 1 and Ham as 0
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    #replacing SMS Data into original place
    raw_text = df[1]
    #2. Text Processing

    processed = raw_text.str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b',
                                     'emailaddr')
    processed = processed.str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',
                                      'httpaddr')
    processed = processed.str.replace(r'£|\$', 'moneysymb')    
    processed = processed.str.replace(
        r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr')    
    processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
    processed = processed.str.replace(r'[^\w\d\s]', ' ')
    processed = processed.str.replace(r'\s+', ' ')
    processed = processed.str.replace(r'^\s+|\s+?$', '')
    processed = processed.str.lower()

    #removing stop words
    processed = processed.apply(lambda x: ' '.join(
        term for term in x.split() if term not in set(stop_words))
    )

    #stemming
    processed = processed.apply(lambda x: ' '.join(
        porter.stem(term) for term in x.split())
    )

    #3. Feature ENgineering
    #3.1 Tokenization - using tf-idf static to create a matrix with each row representing training example and each col 
    #    representing the tf-idf value of nth gram word(here we use 1 gram and 2 gram , i.e taking 1 work at a time , 
    #    and taking two words at a time). This is also called vectorization

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_ngrams = vectorizer.fit_transform(processed)
    #X_ngrams.shape #pretty big matrix with 36348 cols showing 1gram or 2-gram words
    #4. Training and evaluating model
    #SVM: finds a hyperplane that deferentiates these two classes

    #Splitting into and Testing Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_ngrams,
        y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc
    )

    #1. SVM Classifier
    clf = LinearSVC(loss='hinge')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print ("Data has been trained on SVM with Score: " , metrics.f1_score(y_test, y_pred),end='')
    
    return clf,vectorizer

def preprocess_text(messy_string):
    assert(type(messy_string) == str)
    cleaned = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', messy_string)
    cleaned = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr',
                     cleaned)
    cleaned = re.sub(r'£|\$', 'moneysymb', cleaned)
    cleaned = re.sub(
        r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr', cleaned)
    cleaned = re.sub(r'\d+(\.\d+)?', 'numbr', cleaned)
    cleaned = re.sub(r'[^\w\d\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'^\s+|\s+?$', '', cleaned.lower())
    return ' '.join(
        porter.stem(term) 
        for term in cleaned.split()
        if term not in set(stop_words)
    )


def spam_filter(message):
    clf, vectorizer = trainmodel()
    if clf.predict(vectorizer.transform([preprocess_text(message)])):
        return 'spam'
    else:
        return 'not spam'

