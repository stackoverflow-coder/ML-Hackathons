# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 19:27:44 2018

@author: Swaroop
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
import xgboost,string
from sklearn_pandas import DataFrameMapper, cross_val_score

trainDF = pd.read_csv(r'D:\Downloads\train.csv')
testDF = pd.read_csv(r'D:\Downloads\test_nvPHrOx.csv')
trainDF['char_count'] = trainDF['Url'].apply(len)
trainDF['word_count'] = trainDF['Url'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['Url'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
trainDF['title_word_count'] = trainDF['Url'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['Url'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(norm = 'l2',analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features = 5000)
features = tfidf.fit_transform(trainDF.Url)
labels = trainDF.Tag

#mapper = DataFrameMapper([
#     ('UrlVect', None),
#     ('char_count', None),
#     ('word_count', None),
#     ('word_density', None),
#     ('punctuation_count', None),
#     ('title_word_count', None),
#     ('upper_case_word_count', None)
#      ])
#
#features = mapper.fit_transform(trainDF)

X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state = 0)
X_valid_tfidf = tfidf.transform(testDF.Url)

#clf = MultinomialNB().fit(X_train_tfidf, y_train)
clf = LinearSVC().fit(X_train, y_train)
#clf = RandomForestClassifier().fit(X_train_tfidf, y_train)
clf = LogisticRegression().fit(X_train, y_train)
#clf = xgboost.XGBClassifier().fit(X_train_tfidf, y_train)

preds = clf.predict(X_valid_tfidf)

print ("Accuracy:", accuracy_score(y_test, preds))
print ("Precision:", precision_score(y_test, preds))
print (classification_report(y_test, preds))
print (confusion_matrix(y_test, preds))

testDF['Tag']=preds
test = testDF.drop(['Domain','Url'],axis=1)
test.to_csv('predictions_v11_tfidf_ngram_5k_LR.csv', index = False)

trainDF['UrlVect']
