# DO test train split

import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import en_core_web_sm
nlp = en_core_web_sm.load()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score
import re
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV

def preprocess(review):
    review = str(review)
    review = re.sub('[^A-Za-z0-9]+', ' ', review)
    review = review.lower()
    lemmatizer = WordNetLemmatizer()
    review =' '.join([lemmatizer.lemmatize(word) for word in review.split()])
    return review



def splitData(data,flag):
    dlst = []
    if flag == 0:
         df= pd.read_csv(updatedData, sep = '\t', quotechar = None, quoting=3, on_bad_lines = 'skip')
    else:
        df= pd.read_csv(data, sep = '\t', quotechar = None, quoting=3, on_bad_lines = 'skip')
    df.loc[-1] = df.columns
    df.index = df.index + 1
    df.sort_index(inplace = True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    df.columns = ['gitId','dataType','label','review']
    df['review'] = df['review'].apply(preprocess)
    train = df.loc[df['dataType'] == 'training']
    valid = df.loc[df['dataType'] == 'validation']
    test = df.loc[df['dataType'] == 'testing']
    return train,valid,test
   
def extractFeatures(train,valid,test):
    ''' Vecotrizing the data after stopwords have been removed'''
    X_train = train['review'].str.lower()
    X_valid = valid['review'].str.lower()
    X_test = test['review'].str.lower()
    y_train = train['label'].str.lower()
    y_valid = valid['label'].str.lower()
    y_test = test['label'].str.lower()

    
    corpus_train = X_train
    vectorizer_train = CountVectorizer(stop_words = 'english')
    X_train = vectorizer_train.fit_transform(corpus_train)
    corpus_valid = X_valid
    vectorizer_valid = CountVectorizer(stop_words = 'english', vocabulary = vectorizer_train.vocabulary_)
    X_valid = vectorizer_train.transform(corpus_valid)
    corpus_test = X_test
    vectorizer_test = CountVectorizer(stop_words = 'english', vocabulary = vectorizer_train.vocabulary_ )
    X_test = vectorizer_train.transform(corpus_test)
    return X_train,X_valid,X_test,y_train,y_valid,y_test


def training(X_train,y_train,X_test):

    y_train = np.array(y_train)    
    model = MultinomialNB()
    parameters = {'alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)}
    grid_search= GridSearchCV(model, parameters)
    grid_search.fit(X_train,y_train)
    predictions = grid_search.predict(X_test)
    return predictions



def metrics(y_test,predictions):
    return precision_score(y_test, predictions, average='macro'), recall_score(y_test, predictions, average='macro'), f1_score(y_test, predictions, average='macro')

if __name__ == '__main__':
    
    data = "unredactor.tsv"
    updatedData = "https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"
    train,valid,test =  splitData(data,flag = 0)
    X_train, X_valid, X_test, y_train, y_valid, y_test  = extractFeatures(train,valid,test)
    predictions = training(X_train,y_train,X_test)
    pred  = list(sorted(zip(*np.unique(predictions, return_counts=True)), key=lambda x: x[1], reverse=True))
    pred = pred[0:20]
    print(pred)
    precision, recall, f1_score = metrics(y_test,predictions)
    print("Precision: ",precision, "Recall:",recall,"F1 Score:",f1_score)
