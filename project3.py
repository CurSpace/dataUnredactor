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
    train = df.loc[df['dataType'] == 'training']
    valid = df.loc[df['dataType'] == 'validation']
    test = df.loc[df['dataType'] == 'testing']
    return train,valid,test
   # for row in dfx.iterrows():
       # print(row)
# extract features from data to train the model
def preprocess(data,valid,test):
    X_t = data['review']
    X_v = valid['review']
    X_tes = test['review']
    y_train = data['label']
    y_valid = valid['label']
    y_test = test['label']
    stopwords = nlp.Defaults.stop_words
  #  print(data.head())
    X_train = []
    for review in X_t:
        text_tokens = word_tokenize(review)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords]
        ureview = " ".join(tokens_without_sw)
        X_train.append(ureview)
    X_valid = []
    for review in X_v:
        text_tokens = word_tokenize(review)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords]
        ureview = " ".join(tokens_without_sw)
        X_valid.append(ureview)
    X_test = []
    for review in X_tes:
        text_tokens = word_tokenize(review)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords]
        ureview = " ".join(tokens_without_sw)
        X_test.append(ureview)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def extractFeatures(X_train,X_valid,X_test):
    ''' Vecotrizing the data after stopwords have been removed'''
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
    X_t = vectorizer_train.fit_transform(corpus_train)
    corpus_valid = X_valid
    vectorizer_valid = CountVectorizer(stop_words = 'english')
    X_v = vectorizer_valid.fit_transform(corpus_valid)
    corpus_test = X_train
    vectorizer_test = CountVectorizer(stop_words = 'english')
    X_tes = vectorizer_test.fit_transform(corpus_test)
    #features = vectorizer.get_feature_names_out()
    return X_t,X_v,X_tes
# Train model to predict missing names
def training(X_t,y_train,X_tes):

    y_train = np.array(y_train)    
    model = MultinomialNB()
    model.fit(X_t,y_train)
    predictions = model.predict(X_tes)
    return predictions
# Training set is the one that the class made

# Look at 2nd column to see if training,valid or testing data

# 3rd columns is the lable

# 4th colunm is the actual sentence with the blank we have to fill

# Every input has only one redaction

if __name__ == '__main__':
    train,valid,test =  splitData(sys.argv[-1])
  # print("Train Set\n",train.head(),"\nValid Set\n",valid,"\nTest\n",test)
    X_train, X_valid, X_test, y_train, y_valid, y_test = preprocess(train,valid,test)
   # print(len(X_train),len(X_valid),len(X_test))
    X_t,X_v,X_tes = extractFeatures(X_train,X_valid,X_test)
   # print(X_t.shape,X_v.shape,X_tes.shape)

    predictions = training(X_t,y_train,X_tes)
    print(predictions)
    print(len(predictions))
