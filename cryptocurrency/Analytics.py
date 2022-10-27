import sys
sys.path.insert(0,'../')
from TwitterAPICredentials import consumerKey, consumerSecret, accessToken, accessTokenSecret
import tweepy
import datetime
from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
# set the access token and the access token secret
authenticate.set_access_token(accessToken, accessTokenSecret)
# create the API objectitcoin' 
api = tweepy.API(authenticate, wait_on_rate_limit=True)

# Gather the 2000 tweets about Bitcoin and filter out any retweets 'RT'
search_coin = 'Bitcoin' 
search_term = '#' + search_coin + ' -filter:retweets'

#current date and time of a coin request.
date_time = datetime.datetime.now()
current_date_time = str(date_time.year)+str(date_time.month)+str(date_time.day)+str(date_time.hour)+str(date_time.minute)
#30 days before the current date and time of the coin request
date_time = datetime.datetime.today() - datetime.timedelta(days=30)
previous_30_days_date_time = str(date_time.year)+str(date_time.month)+str(date_time.day)+str(date_time.hour)+str(date_time.minute)
# create a cursor object
tweets = []
# tweepy.Cursor(api.search_tweets, q, tweet_mode='extended').items(tweetNumber)
tweets = tweepy.Cursor(api.search_tweets, q=search_term, lang='en', tweet_mode='extended', fromDate = previous_30_days_date_time, toDate = current_date_time).items(500)
# store the tweets in a variable and get the full text

all_tweets = [tweet.full_text for tweet in tweets]

#Create a dataframe to store the tweets with a column called Tweets
df = pd.DataFrame(all_tweets, columns=['Tweets'])

# Create a function to clean the tweets
def cleanTwt(twt):
    twt = re.sub('#bitcoin', 'bitcoin', twt) # Removes the '#' from bitcoin
    twt = re.sub('#Bitcoin', 'Bitcoin', twt) # Removes the '#' from Bitcoin
    twt = re.sub('#[A-Za-z0-9]+', '', twt) # Removes any strings with a '#'
    twt = re.sub('\\n', '', twt) # removes the '\n' string
    twt = re.sub('https?:\/\/\S+', '', twt) # Removes any hyperlinks
    return twt
# Clean the Tweets
df['Cleaned_Tweets'] = df['Tweets'].apply(cleanTwt)
#x = df['Tweets'][0]
#print(x)

# Create a function to get the subjectifvity
def getSubjectivity(twt):
  return TextBlob(twt).sentiment.subjectivity
# Create a function to get the polarity
def getPolarity(twt):
  return TextBlob(twt).sentiment.polarity

#Create two new columns called 'subjectivity' and 'Polarity'
df['Subjectivity'] = df['Cleaned_Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Cleaned_Tweets'].apply(getPolarity)

#Create a function to get the sentiment text
def getSentiment(score):
  if score < 0:
    return 'Negative'
  elif score ==0:
    return 'Neutral'
  else:
    return 'Positive'

#Create a column to score the text sentiment
df['Sentiment'] = df['Polarity'].apply(getSentiment)

#Create a scatter plot to show the subjectivity and polarity 
plt.figure(figsize=(8,6))
for i in range(0, df.shape[0]):
  plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Purple')
plt.title('Sentiment Analysis Scatter Plot')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity(objective -> subjective)')
plt.show()

#Create a bar chart to show the count of Positive, Neutral and Negative sentiments
df['Sentiment'].value_counts().plot(kind='bar')
plt.title('Sentiment Analysis Bar Plot')
plt.xlabel('Sentiment')
plt.ylabel('Number of tweets')
plt.show()