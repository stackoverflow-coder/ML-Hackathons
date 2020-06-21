# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 19:42:14 2018

@author: Swaroop
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn_pandas import CategoricalImputer

train = pd.read_csv(r"D:\Documents\Hackathon - Analytics Vidhya\AMExpert\train.csv")
test = pd.read_csv(r"D:\Documents\Hackathon - Analytics Vidhya\AMExpert\test.csv")

year = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M" ).year
month = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M" ).month
day = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M" ).day
hour = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M" ).hour
minute = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M" ).minute
weekday = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M" ).isoweekday

train['year'] = train['DateTime'].map(year)
train['month'] = train['DateTime'].map(month)
train['day'] = train['DateTime'].map(day)
train['hour'] = train['DateTime'].map(hour)
train['minute'] = train['DateTime'].map(minute)
train['weekday'] = train['DateTime'].map(weekday)

test['year'] = test['DateTime'].map(year)
test['month'] = test['DateTime'].map(month)
test['day'] = test['DateTime'].map(day)
test['hour'] = test['DateTime'].map(hour)
test['minute'] = test['DateTime'].map(minute)
test['weekday'] = test['DateTime'].map(weekday)

train = train.astype(str)
test = test.astype(str)


cols = ["user_id","product","campaign_id","webpage_id","product_category_1","product_category_2"
        ,"user_group_id","gender","age_level","city_development_index","var_1"]


labelencoder = LabelEncoder()
for i in cols:
    train[i] = labelencoder.fit_transform(train[i])

X = train[cols]
Y = train["is_click"].astype(int).values


params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',silent=True, nthread=4)
xgb.fit(X, Y)

rf = RandomForestClassifier(n_estimators=600)
rf.fit(X,Y)


folds = 3
param_comb = 3

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, 
                                   scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001 )
random_search.fit(X, Y)


#grid = GridSearchCV(estimator=xgb, param_grid=params, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3 )
#grid.fit(X, Y)


X_test = test
for i in cols:
    X_test[i] = labelencoder.fit_transform(test[i])

X_test["is_click"] = rf.predict(X_test[cols])

submission = X_test[["session_id","is_click"]]

submission.to_csv("RF.csv",index=False)


