# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


train  = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


def int_to_cat(df,col_names):
    for i in col_names:
        df[i] = df[i].astype('category')
    return df
   

train.dtypes
train.describe()
train.isna().sum(axis=0)
    
colnames = ['employee_id','no_of_trainings','previous_year_rating','length_of_service','KPIs_met >80%','awards_won?','is_promoted']
train = int_to_cat(train,colnames)

impute = Imputer()

train["is_promoted"].value_counts()
train["department"].value_counts()
train["region"].value_counts()
train["education"].value_counts()
train["gender"].value_counts()
train["recruitment_channel"].value_counts()
train["no_of_trainings"].value_counts()
train["previous_year_rating"].value_counts()
train["length_of_service"].value_counts()
train["KPIs_met >80%"].value_counts()
train["awards_won?"].value_counts()
train["previous_year_rating"].value_counts()
train["age"].value_counts()

pd.crosstab(train.is_promoted,train.education)
pd.crosstab(train.is_promoted,train.gender)
pd.crosstab(train.is_promoted,train["KPIs_met >80%"])
pd.crosstab(train.is_promoted,train["awards_won?"])
pd.crosstab(train.is_promoted,train["previous_year_rating"])


X = train.loc[:, train.columns != 'is_promoted']
y = train.loc[:, train.columns == 'is_promoted']

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['is_promoted'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))


logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
