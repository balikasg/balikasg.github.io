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



As in any machine learning pipeline, to demonstrate the effectiveness of machine learning tools we will first apply a feature extraction step and then feed a classification system. Then, we will revisit the feature extraction process to add more features that were shown to be suitable for sentiment analysis and we will evaluate their effect. For demonstration purposes I will be using the data released by the organisers of Task 4 of SemEval-2017 that also comprised various sentiment Twitter classification problems. The data are available [here.](http://alt.qcri.org/semeval2017/task4/?id=download-the-full-training-data-for-semeval-2017-task-4) In the Table below you may find the distribution of the tweets over the categories.

| Category | # of instances |
|:-------------: | :--------------------------: |
| *Negative* | XX |
| *Neutral*  | XX |
| *Positive* | XX |



Without further a-do let's load the data and build a classifier using standard vectorization methods. In this step we will compare the `CountVectorizer`, `TfIdfVectorizer` and `HashingVectorizer` of `sklearn`. 

```python
for vect in [(CountVectorizer( ngram_range=(1,1), analyzer='word', min_df=5, tokenizer=tokenizer.tokenize,), CountVectorizer( ngram_range=(3,5), analyzer='char', min_df=5, tokenizer=tokenizer.tokenize)),
            (TfidfVectorizer( ngram_range=(1,1), analyzer='word', min_df=5, tokenizer=tokenizer.tokenize,), TfidfVectorizer( ngram_range=(3,5), analyzer='char', min_df=5, tokenizer=tokenizer.tokenize))]: 
    my_vect = pipeline.FeatureUnion([("ngram", vect[0]), ("cgram", vect[1])], n_jobs=1)
    X = my_vect.fit_transform(text_train)
    clf = grid_search.GridSearchCV(linear_model.LogisticRegression(class_weight='balanced'), param_grid={"C":[0.01, 0.1, 1, 10]}, cv=3, n_jobs =-1, scoring='recall_macro' )
    clf.fit(X, y_train)
    preds = clf.predict(my_vect.transform(text_test))
    print metrics.recall_score(y_test, preds, average='macro')

```

As you can see from the snippet above...





