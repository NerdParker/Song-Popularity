#!/usr/bin/env python
# coding: utf-8

# In[1]:


# packages to store and manipulate data
import numpy as np
import pandas as pd

# visualization packages
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# ml modeling packages
import copy
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier

from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import SVC, LinearSVC


# In[2]:


# Import Dataset
df = pd.read_csv('C:/Users/607791/Desktop/DS/Practicum/MSD20k_and_BB.csv')
df.head()


# In[3]:


df_clean=df.drop(['artist_id'], axis=1)
df_clean=df_clean.drop(['artist_latitude'], axis=1)
df_clean=df_clean.drop(['artist_longitude'], axis=1)
df_clean=df_clean.drop(['artist_name'], axis=1)
df_clean=df_clean.drop(['artist_location'], axis=1)
df_clean=df_clean.drop(['end_of_fade_in'], axis=1)
df_clean=df_clean.drop(['start_of_fade_out'], axis=1)
df_clean=df_clean.drop(['release'], axis=1)
df_clean=df_clean.drop(['title'], axis=1)
df_clean=df_clean.drop(['year'], axis=1)
df_clean=df_clean.drop(['key_confidence'], axis=1)
df_clean=df_clean.drop(['mode_confidence'], axis=1)
df_clean=df_clean.drop(['time_signature_confidence'], axis=1)


# In[4]:


df.shape


# In[5]:


df_clean=df_clean.dropna()
df_clean.shape


# In[6]:


df_clean.head()


# In[7]:


hotness = copy.deepcopy(df_clean.bb_hotsong)
df_hot = df_clean.drop("bb_hotsong", axis=1)


# In[8]:


# training/test data, test prediction XGB accuracy
x_train, x_test, y_train, y_test = train_test_split(df_hot, hotness, test_size=0.35, random_state=2)
model = XGBClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
accuracy = model.score(x_test, y_test)

print("Prediction Accuracy: %.1f%%" % (accuracy * 100.0))


# In[9]:


# cross validation for each model
def modeling(model, x_train, y_train):
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring = "roc_auc")
    print("Cross Validation Scores:", scores)
    print("Cross Validation Mean:", scores.mean())
    print("Cross Validation Standard Deviation:", scores.std())
    print("Model as Percentage: ", scores.mean()*100)
    return scores.mean()


# In[10]:


#vlogistic regression classifier
log_reg = LogisticRegression()
log_reg.fit(df_hot, hotness)
log_reg_res = modeling(log_reg, df_hot, hotness)


# In[11]:


# knn classifier, best neighbors = 8
k_near_neigh = KNeighborsClassifier(n_neighbors = 10)
k_near_neigh.fit(df_hot, hotness)
k_near_neigh_res = modeling(k_near_neigh, df_hot, hotness)


# In[12]:


# xgboost classifier
x_grad_boost = XGBClassifier(learning_rate =0.2, n_estimators=80, max_depth=6, min_child_weight=1, gamma=0, subsample=0.7,
    colsample_bytree=0.7, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=12)
x_grad_boost.fit(df_hot, hotness)
x_grad_boost_res = modeling(x_grad_boost, df_hot, hotness)


# In[13]:


# random forest classifier
random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(df_hot, hotness)
random_forest_res = modeling(random_forest, df_hot, hotness)


# In[14]:


# decision tree classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(df_hot, hotness)
decision_tree_res = modeling(decision_tree, df_hot, hotness)


# In[15]:


# linear SVC classifier
linear_svc = LinearSVC()
linear_svc.fit(df_hot, hotness)
linear_svc_res = modeling(linear_svc, df_hot, hotness)


# In[16]:


# compare scores
comparison = pd.DataFrame({'Model': ['Logistic Regression','K-Nearest Neighbors','Extreme Gradient Boosting',
            'Random Forest','Decision Tree','Linear SVC'],
    'Score': [log_reg_res,k_near_neigh_res,x_grad_boost_res,random_forest_res,decision_tree_res,linear_svc_res]})
df_comparison = comparison.sort_values(by='Score', ascending=False)
df_comparison = df_comparison.set_index('Score')
df_comparison


# In[ ]:




