---
title:  "Polars is awesome!"
categories: ["Python"]
tags: ["polars"]
mathjax: fasle
---

I recently came across the this excellent [calmcode tutorial about polars](https://calmcode.io/polars/introduction.html) and its [write-up ](https://www.pola.rs/posts/the-expressions-api-in-polars-is-amazing/) and I was intrigued. Time passed but I finally found the time to discover the library and I believe it is amazing! I am very enthousiatic about it for several reasons: 
- It is really fast! It is so fast that it pushes the limits of what can be done in a local machine without access to a Spark cluster. 
- The syntax, although it requires some effort in the beginning, is really intuitive. Actually, if you have prior pySpark or scala Spark experience which not uncommon today, it is much easier to use it and quickly onboard. 
- I found my code to be elegant.. 

In this post I will show a quick demo with the AOL query dataset that can be obtained [from kaggle](https://www.kaggle.com/datasets/dineshydv/aol-user-session-collection-500k). This not a huge dataset, ~3.6M lines in a tsv. Yet, it is enough to show the merits of polars. 

## Installation
Just to show how easy it is to install it:  
```
pip install polars pandas
```  

## Demo-Timings 

I will illsutrate an initial feature generation pipeline, both with pandas and with polars and we will compare the timings. 
After loading the data, we will extract some timeseries features (weekday, month) and some user prevalance stats (Number of queries and number of clicks a user did). Also, we will append a column on the data with the number of times a query occurs in the data. Typically, in pandas you would do groupBy and joins for such features. 
Here we will do one groupBy with Named aggregations and we will also use `transform` to elegantly add columns to the dataframe. 

Here is how the data look like: 
![AOL data head]({{ site.url }}/assets/polars1.png)
From the columns, `AnonID` is a `user_id`, and `ItemRank` and `ClickURL` are the position and the url of the clicked item when someone clicked (NaN otherwise). 

Here is the pandas code:
```python
df = pd.read_csv(path_to_data, delimiter="\t")
df['QueryTime'] = pd.to_datetime(df['QueryTime'])
per_user_queries = df.groupby("AnonID", as_index=False).agg(nbQueries=("Query", "count"), NbClicks=("ClickURL","count"))
df['month'] = df['QueryTime'].dt.month
df['weekday'] = df['QueryTime'].dt.day_of_week
df = df.merge(per_user_queries, on=['AnonID'])
df['QueryOccurencesInData'] = df.groupby('Query')['Query'].transform('size')
```
I like to believe that this is decent pandas code. Running in an iMac n jupyter notebook and timing with `%%time` cell magic returns 6.5sec. 

Here is the polars code: 
```python
df = pl.scan_csv(path_to_data, parse_dates=True, sep="\t")
df.select([
    pl.all(),
    pl.col('Query').count().over('AnonID').alias('nbQueries'),
    pl.col('ClickURL').drop_nulls().count().over('AnonID').alias('PerUserCount'),
    pl.col('QueryTime').dt.month().alias("month"),
    pl.col('QueryTime').dt.weekday().alias("weekday"),
    pl.col('QueryTime').count().over('Query').alias("QueryOccurencesInData")
]).collect().limit(10)
```
They both return ideantical results (except the day of week, which has a delta of 1). The difference is that the polars code takes 860ms!!! *It is around 8x faster!*
What is more, the syntax is really nice for my taste! The way select is used and the way window aggregations can do groupBy's (look the use of `.over`) is very elegant! Notice, that all these aggregations are done within a single select and polars takes care of executing them embarassingly parallel!


## Concluding thoughts 
I loved it taking time with `polars`. I am very enthousiastic about it and I will start using the project more and more. As a matter of fast I used it with a dataset of ~20GB for work. Cleaning and feature generation took around 7 min.. With pandas, I had to split the jobs in smaller tasks with loops and the same task took more than an hour. So this is an actual 10x speed-up out-of-the-box that made my day. 


