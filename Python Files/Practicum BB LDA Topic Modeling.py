#!/usr/bin/env python
# coding: utf-8

# In[1]:


# package to clean text
import nltk; nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re

# packages to store and manipulate data
import numpy as np
import pandas as pd
from pprint import pprint

# gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# spacy for lemmatization
import spacy
nlp = spacy.load("en_core_web_sm")

# plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


# Import Dataset
df = pd.read_csv('C:/Users/607791/Desktop/DS/Practicum/billboard_lyrics_1964-2015.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df=df.drop(['Source'], axis=1)
df=df.dropna()
df.shape


# In[5]:


df.head()


# In[6]:


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


# In[7]:


stopwords = nltk.corpus.stopwords.words('english')
stopwords = set(stopwords + ['use', 'make', 'see', 'how', 'go', 'say', 'ask', 'get'])


# In[8]:


import string
# list

data = df.Lyrics.values.tolist()

data = clean_text(data)


# In[9]:


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


# In[10]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
lyr_words = list(sent_to_words(data))


# In[11]:


# bigram and trigram models
bigram = gensim.models.Phrases(lyr_words, min_count=8, threshold=100)
trigram = gensim.models.Phrases(bigram[lyr_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# output test
print(trigram_mod[bigram_mod[lyr_words[0]]])


# In[12]:


"""https://spacy.io/api/annotation"""

def bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


# In[13]:


# bigrams
lyr_bigrams = bigrams(lyr_words)
lyr_bigrams
# lemmatization
lyr_lemmatized = lemmatization(lyr_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(lyr_lemmatized[:1])


# In[14]:


# Dictionary
id2word = corpora.Dictionary(lyr_lemmatized)
# Corpus
texts = lyr_lemmatized
# lyric term frequency
corpus = [id2word.doc2bow(text) for text in texts]


# In[15]:


# LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=8, 
                                           random_state=10,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[16]:


from pprint import pprint
# Topics key words
pprint(lda_model.print_topics())
lda = lda_model[corpus]


# In[17]:


# Perplexity measures the model with lower scores being the goal and a sign of a good model
print('Perplexity: ', lda_model.log_perplexity(corpus))

# Coherence measures the proposed topics distinguishability
coherence_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
coherence = coherence_lda.get_coherence()
print('Coherence: ', coherence)


# In[18]:


for i in range(len(lyr_lemmatized)):
    lyr_lemmatized[i] = ' '.join(lyr_lemmatized[i])
    
df['clean'] = lyr_lemmatized
df.head()


# In[19]:


import pyLDAvis
import pyLDAvis.gensim 

# Topic Visualization
pyLDAvis.enable_notebook()
distance_map = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
distance_map


# In[20]:


# get topics for each song
topics = lda_model.get_document_topics(corpus, per_word_topics=True)
all_topics = [(doc_topics, word_topics, word_phis) for doc_topics, word_topics, word_phis in topics]
all_topics


# In[21]:


# pull out just the topics from the get_document_topics function
def Extract(all_topics): 
    return [doc[0] for doc in all_topics] 


# In[22]:


# pull out just the topics from the get_document_topics function
# each bracket is a sublist of the score each topic has
all_doc_topics = Extract(all_topics)
df['all_doc_topics'] = Extract(all_doc_topics)
df.head()
all_doc_topics


# In[23]:


# sorting sublist to pull the top topic to the front of each sublist
for sublist in all_doc_topics: 
    sublist.sort(key = lambda x: x[1],reverse=1) 
    
all_doc_topics


# In[24]:


# extract the top topic and add it to the data
def topic(all_doc_topics): 
    to_sort = [[item[1],item[0]] for item in all_doc_topics]
    to_sort.sort(reverse=1)
    max_value,max_topic = to_sort[0][0],to_sort[0][1]
    return [doc[0] for doc in all_doc_topics]

topics = topic(all_doc_topics)

df['topic'] = topics
df.head()


# In[25]:


# separate the topics from their values and capture both
topic_num = [item[0] for item in topics]
df['topic'] = topic_num

topic_value = [item[1] for item in topics]
df['topic_value'] = topic_value

df.head()


# In[26]:


train_vector = []
for i in range(len(data)):
    top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(5)]
    train_vector.append(topic_vec)


# In[27]:


train_vector


# In[29]:


# classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn import metrics
#logistic regression to label unseen data with topics
X = np.array(train_vector)
y = np.array(topic_num)

# kfold to split data with 1/6th for validation
kf = KFold(6, shuffle=True, random_state=10)
log_reg = []

for train, valid in kf.split(X, y):
    X_train, y_train = X[train], y[train]
    X_valid, y_valid = X[valid], y[valid]
    
    scaler = StandardScaler()
    X_trainS = scaler.fit_transform(X_train)
    X_valS = scaler.transform(X_valid)

    # logisitic regression model
    logreg_model = LogisticRegression(class_weight= 'balanced',solver='newton-cg',fit_intercept=True).fit(X_trainS, y_train)

    logreg_y_predict = logreg_model.predict(X_valS)
    log_reg.append(f1_score(y_valid, logreg_y_predict, average=None))


# In[30]:


# logistic regression result with 1.0 being 100% success and a standard deviation
print(f'Logistic Regression Result: {np.mean(log_reg):.3f} +- {np.std(log_reg):.3f}')


# In[32]:


import scipy
corpus_umap = gensim.matutils.corpus2csc(corpus).T
# normalize by row
corpus_umap = corpus_umap.multiply(scipy.sparse.csr_matrix(1/np.sqrt(corpus_umap.multiply(corpus_umap).sum(1))))
# double check the norms
np.sum(np.abs(corpus_umap.multiply(corpus_umap).sum(1) - 1) > 0.001)


# In[33]:


# dimension Reduction using UMAP
import umap.umap_ as umap
reducer = umap.UMAP()
#%%time
embedding = umap.UMAP(metric="cosine", n_components=2).fit_transform(corpus_umap)


# In[34]:


def get_model_results(ldamodel, corpus, id2word):
    vis = pyLDAvis.gensim.prepare(ldamodel, corpus, id2word, sort_topics=False)
    transformed = ldamodel.get_document_topics(corpus)
    df = pd.DataFrame.from_records([{v:k for v, k in row} for row in transformed])
    return vis, df  


# In[37]:


lda_vis, lda_result  = get_model_results(lda_model, corpus, id2word)


# In[38]:


lda_article = df[['Song', 'clean', 'topic']]


# In[39]:


df_emb = pd.DataFrame(embedding, columns=["x", "y"])
df_emb.head()


# In[43]:


df_emb["label"] = df[['topic']]
df_emb.head()
df_emb.shape


# In[44]:


df_emb=df_emb.dropna()
df_emb.shape


# In[57]:


#TF-IDF matrix embedded in two dimensions by UMAP
df_emb_sample = df_emb.sample(2000)
fig, ax = plt.subplots(figsize=(14, 14/1.5))
plt.scatter(
    df_emb_sample["x"].values, df_emb_sample["y"].values, s=2, c=df_emb_sample["label"].values
)
plt.setp(ax, xticks=[], yticks=[])
colorbar = plt.colorbar(boundaries=np.arange(8)-0.5)
colorbar.set_ticks(np.arange(7))
plt.title("Song Topics", fontsize=20)
plt.show()


# In[62]:


# see topics individually
g = sns.FacetGrid(df_emb, col="label", col_wrap=2, height=5, aspect=1)
g.map(plt.scatter, "x", "y", s=0.2).fig.subplots_adjust(wspace=.05, hspace=.5)

# 53% of songs fall into topic 1
#  '0.053*"know" + 0.052*"love" + 0.032*"baby" + 0.027*"want" + 0.020*"feel" + '
#  '0.020*"time" + 0.020*"never" + 0.018*"tell" + 0.018*"would" + '
#  '0.016*"think"'


# In[ ]:




