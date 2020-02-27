# Song-Popularity-Capstone
This capstone project serves to answer the question of what makes a song popular or successful with an increased focus on lyrics and the top 100 billboard songs from 1964-2015.
### Contents:
1. Data Exploration and Cleaning
2. Lyric Sentiment Analysis
3. Billboard Toppers LDA Topic Modeling and UMAP Clustering
4. Million-song Subset Machine Learning Models
5. Conclusions
6. Future Work

### Data Exploration and Cleaning:
All of the data files can be found in the "Data" folder.
The initial data cleaning and exploration can be found in `Practicum BB Lyric Sentiemnt Analysis.ipynb` and `Practicum MS subset Exploration.ipynb` in the `Jupyter Notebooks` folder as well as the respective python files in the `Python files` folder.

1. The top 100 billboard song lyrics data from 1964 - 2015 after some general cleaning looks like this:
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Lyric%20Dataset%20Sample.PNG)

2. Next I cleaned the lyrics by removing punctuation and digits. I left the pronouns, lemmatized the rest to remove the stop words and then joined them back together. I also removed any links, emails and quotes out of habit. 
A wordcloud of the results is below:
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Cleaned%20Lyric%20WordCloud.PNG)

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Top%20Words%20Visual.PNG)

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Artists%20With%20most%20BB%20Toppers.PNG)

### Lyric Sentiment Analysis:

### Billboard Toppers LDA Topic Modeling and UMAP Clustering:

### Million-song Subset Machine Learning Models:

### Conclusions:

### Future Work:
