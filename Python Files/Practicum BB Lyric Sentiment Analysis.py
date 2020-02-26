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


df.head()


# In[11]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
lyr_words = list(sent_to_words(data))


# In[12]:


# bigram and trigram models
bigram = gensim.models.Phrases(lyr_words, min_count=8, threshold=100)
trigram = gensim.models.Phrases(bigram[lyr_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# output test
print(trigram_mod[bigram_mod[lyr_words[0]]])


# In[13]:


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


# In[14]:


# bigrams
lyr_bigrams = bigrams(lyr_words)
lyr_bigrams
# lemmatization
lyr_lemmatized = lemmatization(lyr_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(lyr_lemmatized[:1])


# In[15]:


for i in range(len(lyr_lemmatized)):
    lyr_lemmatized[i] = ' '.join(lyr_lemmatized[i])
    
df['clean'] = lyr_lemmatized
df.head()


# In[16]:


cleaned = df.clean.to_string()


# In[17]:


# overall sentiment of all lyrics
# textblob uses a lookup dictionary for sentiment and subjectivity 
from textblob import TextBlob
TextBlob(cleaned).sentiment


# In[18]:


from nltk.corpus import subjectivity
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[19]:


import textblob
tb = textblob.TextBlob(df.clean[0])
tb.sentiment_assessments


# In[20]:


# naive bayes sentiment classification, sentiment probabilities
nb = textblob.en.sentiments.NaiveBayesAnalyzer()
nb.analyze(df.clean[0])


# In[21]:


nb.analyze(df.clean[1])


# In[22]:


# rule based method for sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[23]:


analyzer.polarity_scores(df.clean[0])


# In[24]:


def sentiment_score(clean_lyric):
    score = analyzer.polarity_scores(clean_lyric)
    weight = score['compound']
    if weight >= 0.1:
        return 1
    elif (weight < 0.1) and (weight > -0.1):
        return 0
    else:
        return -1


# In[25]:


sentiment_score(df.clean[0])


# In[26]:


sent = [TextBlob(Lyrics) for Lyrics in clean_lyric]
sent[0].polarity, sent[0]

val_sentiment = [[Lyrics.sentiment.polarity, str(Lyrics)] for Lyrics in sent]
val_sentiment[0]

df_sentiment = pd.DataFrame(val_sentiment, columns=["polarity", "clean_lyric"])
df_sentiment.head()


# In[27]:


from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(10, 7))

# polarity histogram
df_sentiment.hist(bins=12,ax=ax)

plt.title("Sentiments of Lyrics")
plt.xlabel("Sentiment Score")
plt.ylabel("Number of Lyrics")
plt.show()


# In[28]:


from wordcloud import WordCloud
long_string = ','.join(list(df.clean))
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[29]:


df_sentiment.sort_values(by=['polarity']).head()


# In[30]:


df_sentiment.sort_values(by=['polarity'],ascending=False).head()


# In[ ]:




