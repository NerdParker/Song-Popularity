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
The initial data cleaning and exploration can be found in `Practicum BB Lyric Sentiment Analysis.ipynb` and `Practicum MS subset Exploration.ipynb` in the `Jupyter Notebooks` folder as well as the respective python files in the `Python files` folder.

1. The top 100 billboard song lyrics data from 1964 - 2015 after some general cleaning looks like this:
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Lyric%20Dataset%20Sample.PNG)

2. Next I cleaned the lyrics by removing punctuation and digits. I left the pronouns, lemmatized the rest to remove the stop words and then joined them back together. I also removed any links, emails and quotes out of habit. 
A wordcloud of the results is below:

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Cleaned%20Lyric%20WordCloud.PNG)

I used matplotlib and seaborn to visualize the top occurring words post cleaning:

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Top%20Words%20Visual.PNG)

I visualized the top artists of the past 50 years with the most billboard toppers:

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Artists%20With%20most%20BB%20Toppers.PNG)

My final initial exploration of the billboard lyrics was to see how lyric variety has changed overtime:

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Lyric%20Variety%20Overtime.PNG)


3. The other major dataset used in this project comes from the Million Song dataset. Obviously, this data would require much more capabilities to handle than I personally possess. So instead I have used a 10,000-song subset of this dataset which contains a variety of useful information about the songs. I combined the billboard lyrics data into the same excel file as this one and used a quick excel lookup to find all the billboard hits in our subset and mark them in a new column called `bb_hotsong`. I removed some data I was not interested in such as artist, song title, year, latitude, longitude and location. I also removed all the rows with missing data and ended up with 5648 songs with 410 appearing on our billboard chart toppers! The result can be seen here:
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/Lyric%20Dataset%20Sample.PNG)

4. In exploring this data the first thing I wanted to do was see if there was any low hanging fruit as far as correlation go so of course I made a correlation plot!

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/MS%20subset%20Variable%20Correlation.PNG)

An alternative way to try to see any data trends although an eyesore is a pairwise plot:

![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/MS%20SNS%20Pairplot.png)

Finally, I examined some of the distributions of other interesting variables:
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/MS%20End%20of%20Fade%20In.png)
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/MS%20Key.png)
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/MS%20Loudness.png)
![alt text](https://github.com/NerdParker/Song-Popularity-Capstone/blob/master/Images/MS%20Tempo.png)



### Lyric Sentiment Analysis:

### Billboard Toppers LDA Topic Modeling and UMAP Clustering:

### Million-song Subset Machine Learning Models:

### Conclusions:

### Future Work:
