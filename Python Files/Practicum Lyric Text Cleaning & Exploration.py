#!/usr/bin/env python
# coding: utf-8

# In[9]:


# package to clean text
import nltk; nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re

# packages to store and manipulate data
import numpy as np
import pandas as pd
from pprint import pprint

# spacy for lemmatization
import spacy
nlp = spacy.load("en_core_web_sm")


# In[10]:


# Import Dataset
df = pd.read_csv('C:/Users/607791/Desktop/DS/Practicum/billboard_lyrics_1964-2015.csv')
df.head()


# In[11]:


df.shape


# In[12]:


df=df.drop(['Source'], axis=1)
df=df.dropna()
df.shape


# In[13]:


df.head()


# In[14]:


# clean text function
def clean_text(docs):
    # remove punctuation and numbers
    print('removing punctuation and digits')
    table = str.maketrans({key: None for key in string.punctuation + string.digits})
    clean_docs = [d.translate(table) for d in docs]
    
    print('spacy nlp...')
    nlp_docs = [nlp(d) for d in clean_docs]
    
    # pronouns stay, rest lemmatized
    print('getting lemmas')
    lemmatized_docs = [[w.lemma_ if w.lemma_ != '-PRON-'
                           else w.lower_
                           for w in d]
                      for d in nlp_docs]
    
    # remove stopwords
    print('removing stopwords')
    lemmatized_docs = [[lemma for lemma in doc if lemma not in stopwords] for doc in lemmatized_docs]
    
    # join tokens back into doc
    clean_docs = [' '.join(l) for l in lemmatized_docs]
        
    return clean_docs


# In[15]:


stopwords = nltk.corpus.stopwords.words('english')
stopwords = set(stopwords + ['use', 'make', 'see', 'how', 'go', 'say', 'ask', 'get'])


# In[16]:


import string
# list

data = df.Lyrics.values.tolist()

data = clean_text(data)


# In[17]:


import re
# remove http links
data = [re.sub('http://\S+', '', sent) for sent in data]

# remove https links
data = [re.sub('https://\S+', '', sent) for sent in data]

# remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# remove single quotes
data = [re.sub("\'", "", sent) for sent in data]

clean_lyric = data


# In[18]:


# wordcloud
from wordcloud import WordCloud
long_string = ','.join(list(data))
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# visualize the top occuring words post cleaning
def top_words(count, vectorize):
    import matplotlib.pyplot as plt
    word = vectorize.get_feature_names()
    total = np.zeros(len(word))
    for t in count:
        total+=t.toarray()[0]
    
    dict_count = (zip(word, total))
    dict_count = sorted(dict_count, key=lambda x:x[1], reverse=True)[0:12]
    word = [w[0] for w in dict_count]
    counts = [w[1] for w in dict_count]
    x_pos = np.arange(len(word)) 
    
    plt.figure(2, figsize=(14, 14/1.5))
    plt.subplot(title='Top Words')
    sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='colorblind')
    plt.xticks(x_pos, word, rotation=90) 
    plt.xlabel('Word')
    plt.ylabel('Total')
    plt.show()

vectorize = CountVectorizer(stop_words='english')
count = vectorize.fit_transform(data)
top_words(count, vectorize)


# In[20]:


top_artist = df.Artist.value_counts()[:25]
top_artist


# In[25]:


plt.figure(2, figsize=(14, 14/1.5))
plt.title("Artist Hits")
df['Artist'].value_counts()[:25].plot('bar')


# In[35]:


df['lyric_total'] = df['Lyrics'].str.split(" ").str.len()
decade_lyrics = df.groupby(['Year'])['lyric_total'].sum()
plt.figure(figsize=(14, 14/1.5))
plt.title("Lyric Variety Overtime",fontsize=25)
decade_lyrics.plot(kind='line')


# In[ ]:




