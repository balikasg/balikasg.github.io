---
title:  "Twitter Sentiment Classification: A Comparative Study"
categories: ["Machine Learning", "Data Science"]
tags: ["text classification", "sentiment analysis"]
mathjax: true
---


# Twitter Sentiment Classification

In this series of posts I will discuss the task of sentiment classification of tweets. Sentiment analysis, as a subfield of opinion mining, has received a lot of attention lately, mainly due to the wide range of applications it enables. Understanding how people feel about different ideas or products for example, can be valuable for marketing purposes. At the same time, the problem is interesting from a research point-of-view as people tend to exress their sentiment and opinions in various ways, often using complex linguistic phenomena like irony or humor. What is more, moving from well structured data like paragraph-sized Amazon reviews to tweets  posses extra challenges as the latter are very short and use creative language, lots of abbreviations etc. For such reasons and also due to the representativeness and easy of data access of Twitter, the subject is very popular. 

I first dealt with sentiment classification in the framework of Task 4 of the SemEval-2016 challenges. The challenge was given a set of tweets to perform different classification tasks like binary classification (Positive/Negative), ternary classification (Positive, Neutral, Negative) or fine-grained classification (VeryNegative, Negative, Neutral, Positive, VeryPositive). In this blogpost I will focus on the ternary sentiment classification task as it is very popular. The Table below shows some example tweets for each sentiment category. Notice the twitter-specific language like mentions @someone, emoticons :q etc. which make the analysis and classification step  challenging. 

| Category | Example |
|:-------------: | :--------------------------: |
| *Negative* |     @Microsoft Heard you are a software company. Why then is most of your software so bad that it has to be replaced by 3rd party apps? |
| *Neutral*  |     @ProfessorF @gilwuvsyou @Microsoft @LivioDeLaCruz We already knew the media march in ideological lockstep but it is nice of him to show it.|
| *Positive*  |     PAX Prime Thursday is overloaded for me with @Microsoft and Nintendo indie events going down. Also, cider!!! :p |
Table: Examples of tweets for each category


As in any machine learning pipeline, to demonstrate the effectiveness of machine learning tools we will first apply a feature extraction step and then feed a classification system. Then, we will revisit the feature extraction process to add more features that were shown to be suitable for sentiment analysis and we will evaluate their effect. For demonstration purposes I will be using the data released by the organisers of Task 4 of SemEval-2017 that also comprised various sentiment Twitter classification problems. The data are available [here.](http://alt.qcri.org/semeval2017/task4/?id=download-the-full-training-data-for-semeval-2017-task-4) In the Table below you may find the distribution of the tweets over the categories.

| Category | # of instances |
|:-------------: | :--------------------------: |
| *Negative* |
| *Neutral*  |
| *Positive* |
Table: Distribution of tweets over the sentiment categories





