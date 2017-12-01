---
title:  "Accelerating tweet downloads with multiple applications and Tweepy"
categories: ["Software", "Data Science"]
tags: ["Twitter", "data collection"]
mathjax: true
---


Social networks and micro-blogging platforms have greatly increased the information access worldwide. At the same time, several research problems have arisen, to study how people generate content and influence others. Among others, Twitter is very popular for research purposes due to its wide adoption, dynamic content and ease of access to its data via the APIs it offers. Downloading tweets to perform various studies is common (e.g., for sentiment analysis studies), but it is impacted by the Twitter user terms and rates imposed when using the APIs. In this post, I am using 
[Tweepy](http://www.tweepy.org/ "Tweepy homepage") to show how given a list of tweet ids, one can download the tweets and accelerate the process by using more than a single application. 

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>


# Problem Definition
Given a list of $N$ tweet ids $l=[id_1, \ldots, id_N]$, we want to download the tweets that correspond to the ids. To do that, we will use Tweepy here, which is a nice framework for Python to access the APIs of Twitter. For simplicity, our output will be just the text of a tweet. The APIs return much more information for a tweet and one can easily use any parts of it accordingly. 

# Code
Lets start by some basic imports and by setting the authentication information of our application that will be used for our calls to the APIs:

```python
import numpy as np, tweepy, codecs, time

consumerKey_1='Key_of_app_1' 
consumerSecret_1='Secret_of_app_1'
token_1='Token_of_app_1'
tokenSecret_1='Secret_of_app_1'

consumerKey_2='Key_of_app_2' 
consumerSecret_2='Secret_of_app_2'
token_2='Token_of_app_2'
tokenSecret_2='Secret_of_app_2'
```
The keys and tokens of your application are available at [Twitter application center](https://apps.twitter.com "Twitter application center"). Having filled this boiler-plate code, the next step is to perform the authentication by calling the Twitter APIs:

```python
auth = tweepy.OAuthHandler(consumerKey_1, consumerSecret_1)
auth.set_access_token(token_1, tokenSecret_1)
api = tweepy.API(auth)

auth2 = tweepy.OAuthHandler(consumerKey_2, consumerSecret_2)
auth2.set_access_token(token_2, tokenSecret_2)
api2 = tweepy.API(auth2)

tweet_ids = open("./your_list_of_ids.txt").read().splitlines() #Loads a list of tweet ids from a file where a line contains an id
tweet_ids = [int(tmp_id) for tmp_id in tweet_ids] # Makes ids integers
```

At this point we have completed the authentication process, the there is a list `tweet_ids` that contains the identifiers of the tweets to be downloaded. 
To respect the Tweeter rates for this case (900 tweets per hour), we will be taking chunks from the `tweet_ids` list of size $Apps \times 900$ (here 1800), downloading them and sleeping for an hour before obtaining the next chunk. Tweepy has a convenient call that returns the allowed rates: 

```python
my_rates = api.rate_limit_status()
print my_rates["resources"]["statuses"]['/statuses/show/:id']
# returns {u'reset': 1512124497, u'limit': 900, u'remaining': 900}
```
Here, `my_rates` is again a json object with much information, but we have isolated only what is needed for our task here.
Below is the function that we will be using for obtaining chunks from the big list with ids: 

```python
def chunks(my_list, chunk_size):
    """Yields successive chunk-sized chunks from my_list."""
    for i in xrange(0, len(my_list), chunk_size):
        yield my_list[i:i + chunk_size]
```
All that remains is to actually perform the downloads and save the results. For simplicity, we will be saving the results to a file, where each line will contain a tweet. 

```python
cnt = 1 # It counts the chunks to print an informative message.
for current_chunk in chunks(tweet_ids, 1800): # Obtain a chunk from our list of tweets
    print("Downloading chunk %d"%cnt), # Informative message..
    cnt += 1
    downloaded_tweets = []
    first_batch, second_batch = current_chunk[:900], current_chunk[900:]
    for my_current_api, batch in zip([api, api2], [first_batch, second_batch]): #(A)
        for tmp_id in batch:
            try: # use try/except to avoid any errors like restricted access, deleted tweet etc..
                tweet = my_current_api.get_status(tmp_id).text # Just keep the text of the returned object for simplicity
                downloaded_tweets.append(tweet)
            except:
                pass 
    with codecs.open("my_downloaded_tweets.txt", 'a', encoding='utf8') as out: # append to the file
        for tweet in downloaded_tweets:
            out.write("%s\n"%tweet)
    print("Done, going to sleep.")
    time.sleep(60*60)
```

This will download the tweets needed. The nice point here, apart from the simplicity of the code, lies at (A). At that loop, we are using several applications to accelerate the download process imposed by the Twitter rates that limit our access to the data.  In case of more applications, there are parts for the code that can be improved to avoid hard-coding, but the purpose here is mainly  to show the logic.

I hope you enjoyed the post! Feel free to comment of tweet about it!

