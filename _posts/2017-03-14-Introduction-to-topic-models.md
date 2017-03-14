---
title:  "Introduction to topic models"
categories: ["Machine Learning", "Data Science"]
tags: ["topic modeling", "unsupervised learning"]
---


# The concept of topic models
The most known topic model, which is the one that I used throughout my thesis as well, is called Latent Dirichlet Allocation (LDA). A topic model is an algorithm that describes an iterative process, whose goal is to uncover the latent themes ("topics" in the topic modeling jargon) that are assumed to generate the words of a document collection. In this post I present some basic concepts of topic modeling. 

The first question the above definition raises concerns these themes/topics: they are groups of words that tend to be semantically coherent and have different probabilities of appearance in text passages that discuss those topics. Imagine, for instance, a topic "Sports" and a topic "Chemistry". Words like "ball", "team" and "score" should have high probability of appearance when the topic is "Sports" and low probability when the topic is "Chemistry". On the other hand, words like "mercury", and "arsenic" should have high probability of appearance when the topic is "Chemistry". The tricky part here is that words do not belong to only a single topic, they belong to every topic, but their probabilities are quite different. For some topics a particular word may appear with very high probability while for other with very low.



Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll's GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
