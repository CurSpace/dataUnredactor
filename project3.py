# DO test train split
import sys
import pandas as pd
import glob
import io
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

def preprocess(review):
    review = str(review)
    review = re.sub('[^A-Za-z0-9]+', ' ', review)
    lemmatizer = WordNetLemmatizer()
   # review = review.replace('.',' ')
   # review = review.replace(',',' ')
    review =' '.join([lemmatizer.lemmatize(word) for word in review.split()])
    return review



def splitData(glob_text):
    dlst = []
    df= pd.read_csv(glob_text, sep = '\t')
    df.loc[-1] = df.columns
    df.index = df.index + 1
    df.sort_index(inplace = True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    df.columns = ['gitId','dataType','label','review']
    df['review'] = df['review'].apply(preprocess)
    print(df.head())
    train = df.loc[df['dataType'] == 'training']
    valid = df.loc[df['dataType'] == 'validation']
    test = df.loc[df['dataType'] == 'testing']
    return train,valid,test
   # for row in dfx.iterrows():
       # print(row)
# extract features from data to train the model



   
# def preprocess(data,valid,test):
#    X_t = data['review']
#    X_v = valid['review']
#    X_tes = test['review']
#    y_train = data['label']
#    y_valid = valid['label']
#    y_test = test['label']

#    stopwords = nlp.Defaults.stop_words
  #  print(data.head())
#    X_train = []
#    for review in X_t:
#        text_tokens = word_tokenize(review)
#        tokens_without_sw = [word for word in text_tokens if not word in stopwords]
#        ureview = " ".join(tokens_without_sw)
#        X_train.append(ureview)
#    X_valid = []
#    for review in X_v:
#        text_tokens = word_tokenize(review)
#        tokens_without_sw = [word for word in text_tokens if not word in stopwords]
#        ureview = " ".join(tokens_without_sw)
#        X_valid.append(ureview)
#     X_test = []
#    for review in X_tes:
#        text_tokens = word_tokenize(review)
#        tokens_without_sw = [word for word in text_tokens if not word in stopwords]
#        ureview = " ".join(tokens_without_sw)
#        X_test.append(ureview)
#    return X_train, X_valid, X_test, y_train, y_valid, y_test

# get features using count and dictionary vectorizer and combine them
def extractFeatures(train,valid,test):
    ''' Vecotrizing the data after stopwords have been removed'''
    X_train = train['review']
    X_valid = valid['review']
    X_test = test['review']
    y_train = train['label']
    y_valid = valid['label']
    y_test = test['label']
    # corpus_train = X_train
    # vectorizer_train = TfidfVectorizer(max_features = 2000)
    # X_t = vectorizer_train.fit_transform(corpus_train)
    # corpus_valid = X_valid
    # vectorizer_valid = TfidfVectorizer(max_features = 2000)
    # X_v = vectorizer_valid.fit_transform(corpus_valid)
    # corpus_test = X_train
    # vectorizer_test = TfidfVectorizer(max_features =2000)
    # X_tes = vectorizer_test.fit_transform(corpus_test)

    corpus_train = X_train
    vectorizer_train = CountVectorizer(stop_words = 'english')
    X_train = vectorizer_train.fit_transform(corpus_train)
    corpus_valid = X_valid
    vectorizer_valid = CountVectorizer(stop_words = 'english', vocabulary = vectorizer_train.vocabulary_)
    X_valid = vectorizer_valid.fit_transform(corpus_valid)
    corpus_test = X_test
    vectorizer_test = CountVectorizer(stop_words = 'english', vocabulary = vectorizer_train.vocabulary_ )
    X_test = vectorizer_test.fit_transform(corpus_test)
    return X_train,X_valid,X_test,y_train,y_valid,y_test

# Train model to predict missing names

# Normalize the data
# Do lemmetization
# Dictionary vectorizer gives bad results
# The merging of vectorizer dose not 
def training(X_train,y_train,X_valid):

    y_train = np.array(y_train)    
    model = MultinomialNB()
    model.fit(X_train,y_train)
    predictions = model.predict(X_valid)
    return predictions
# Training set is the one that the class made

# Look at 2nd column to see if training,valid or testing data

# 3rd columns is the lable

# 4th colunm is the actual sentence with the blank we have to fill

# Every input has only one redaction

# calculate scores
def metrics(y_train,predictions):
    return precision_score(y_valid, predictions, average='macro'), recall_score(y_valid, predictions, average='macro'), f1_score(y_valid, predictions, average='macro')

if __name__ == '__main__':
    train,valid,test =  splitData(sys.argv[-1])
    X_train, X_valid, X_test, y_train, y_valid, y_test  = extractFeatures(train,valid,test)

    print(X_train.shape,X_valid.shape,X_test.shape)
    predictions = training(X_train,y_train,X_valid)
    print(predictions)
    print(len(predictions))
    precision, accuracy, recall = metrics(y_train,predictions)
    print(precision, accuracy, recall)
