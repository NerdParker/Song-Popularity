#!/usr/bin/env python
# coding: utf-8

# In[1]:


# packages to store and manipulate data
import numpy as np
import pandas as pd
from pprint import pprint

# spacy for lemmatization
import spacy
nlp = spacy.load("en_core_web_sm")

# packages for visualizations
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Import Dataset
df = pd.read_csv('C:/Users/607791/Desktop/DS/Practicum/MSD20k_and_BB.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df_clean=df.drop(['artist_id'], axis=1)
df_clean=df_clean.drop(['artist_latitude'], axis=1)
df_clean=df_clean.drop(['artist_location'], axis=1)
df_clean=df_clean.drop(['artist_longitude'], axis=1)
df_clean=df_clean.drop(['artist_name'], axis=1)
df_clean=df_clean.drop(['key_confidence'], axis=1)
df_clean=df_clean.drop(['mode_confidence'], axis=1)
df_clean=df_clean.drop(['release'], axis=1)
df_clean=df_clean.drop(['time_signature_confidence'], axis=1)
df_clean=df_clean.drop(['title'], axis=1)
df_clean=df_clean.drop(['year'], axis=1)
df_clean=df_clean.dropna()
df_clean.shape


# In[5]:


df_clean.head()


# In[6]:


# initial correlation plot
correlation = df_clean.corr()
axis = sns.heatmap(
    correlation, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(30, 230, n=200)
)
axis.set_xticklabels(
    axis.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

# notable correlations of song hotness are artist hotness/familiarity which make 
# sense and to a lesser degree loudness.


# In[7]:


# variable pair plot
sns.pairplot(df_clean);


# In[8]:


def distrib(col):
    graph = sns.kdeplot(df_clean[col][(df_clean["bb_hotsong"] == 1)], color="Gray", shade = True)
    graph.set_xlabel(col)
    graph.set_ylabel("Frequency")
    plt.show()


# In[9]:


distrib("end_of_fade_in")


# In[10]:


distrib("key")


# In[11]:


distrib("loudness")


# In[12]:


distrib("start_of_fade_out")


# In[13]:


distrib("tempo")


# In[ ]:




