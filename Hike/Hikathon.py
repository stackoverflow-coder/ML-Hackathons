# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:50:39 2019

@author: Swaroop
"""

import pandas as pd

train = pd.read_csv('train.csv')
train.head(5)
train.dtypes


user_features = pd.read_csv('user_features.csv')
user_features.head(5)
user_features.dtypes

train['is_chat'].value_counts()

user_features[user_features['node_id'] == 8446602]
user_features[user_features['node_id'] == 6636127]


train[train['is_chat']==1].head(2)

user_features[user_features['node_id'] == 7159649]
user_features[user_features['node_id'] == 7791327]


user_features[user_features['node_id'] == 4771042]
user_features[user_features['node_id'] == 998845]


sum_f1 = 