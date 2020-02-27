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
The initial data cleaning and exploration can be found in `Practicum Lyric Text Cleaning & Exploration.ipynb` and `Practicum MS subset Exploration.ipynb` in the `Jupyter Notebooks` folder as well as the respective python files in the `Python files` folder.

1. The top 100 billboard song lyrics data from 1964 - 2015 after some general cleaning looks like this:
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Lyric%20Dataset%20Sample.PNG)

2. Next I cleaned the lyrics by removing punctuation and digits. I left the pronouns, lemmatized the rest to remove the stop words and then joined them back together. I also removed any links, emails and quotes out of habit. 
A wordcloud of the results is below:

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Cleaned%20Lyric%20WordCloud.PNG)

The wordcloud reveals that the top words are love, know, cause, want, oh oh, baby, girl, come etc. We can see this further with a bar chart. 
I used matplotlib and seaborn to visualize the top occurring words post cleaning:

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Top%20Words%20Visual.PNG)

I visualized the top artists of the past 50 years with the most billboard toppers:

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Artists%20With%20most%20BB%20Toppers.PNG)

We can see that there are many artists who are great and quite close together in chart toppers in the top artists but no one comes close to Madonna.
My final initial exploration of the billboard lyrics was to see how lyric variety has changed overtime:

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Lyric%20Variety%20Overtime.PNG)

Lyric variety appears to have peaked in the early 2000s and has generally been on the decline. Interestingly the lyric variety is still double what it was around the 1970s.

3. The other major dataset used in this project comes from the Million Song dataset. Obviously, this data would require much more capabilities to handle than I personally possess. So instead I have used a 10,000-song subset of this dataset which contains a variety of useful information about the songs. I combined the billboard lyrics data into the same excel file as this one and used a quick excel lookup to find all the billboard hits in our subset and mark them in a new column called `bb_hotsong`. I removed some data I was not interested in such as artist, song title, year, latitude, longitude and location. I also removed all the rows with missing data and ended up with 5648 songs with 410 appearing on our billboard chart toppers! The result can be seen here:
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Lyric%20Dataset%20Sample.PNG)

4. In exploring this data the first thing I wanted to do was see if there was any low hanging fruit as far as correlation go so of course I made a correlation plot!

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/MS%20subset%20Variable%20Correlation.PNG)

Unfortunately, none of the data appears to be correlated with billboard hotness, this may be in part to only having data for 410/5100 of our billboard toppers. We do see a slight correlation between song hotness and artist hotness/familiarity. This suggests that already popular/known artists have a higher likelihood of having popular songs, but it is no means guaranteed.

An alternative way to try to see any data trends although an eyesore is a pairwise plot:

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/MS%20SNS%20Pairplot.png)

Again, not much happening with billboard hotness but the usefulness of the pairwise plot can be seen quite clearly in some variables such as the correlation between song duration and start of fade out. 

Finally, I examined some of the distributions of other interesting variables:
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/MS%20End%20of%20Fade%20In.png)
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/MS%20Key.png)
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/MS%20Loudness.png)
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/MS%20Tempo.png)

Not a whole lot interesting to be derived from this, given the time I would like to come back and compare these figures with identical ones that only incorporate the billboard toppers to perhaps notice a trend in the billboard hits although our correlation plot suggests that might be pointless.


### Lyric Sentiment Analysis:
The lyric sentiment analysis work can be found in `Practicum BB Lyric Sentiment Analysis.ipynb` in the `Jupyter Notebooks` folder as well as the respective python files in the `Python files` folder.

### Billboard Toppers LDA Topic Modeling and UMAP Clustering:
The LDA Topic Modeling and UMAP Clustering work can be found in `Practicum BB LDA Topic Modeling.ipynb` in the `Jupyter Notebooks` folder as well as the respective python files in the `Python files` folder.

### Million-song Subset Machine Learning Models:
The million-song subset machine learning models work can be found in `Practicum MSD ML Models.ipynb` in the `Jupyter Notebooks` folder as well as the respective python files in the `Python files` folder.

### Conclusions:

### Future Work:
