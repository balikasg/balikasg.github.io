---
title:  "Introduction to topic models"
categories: ["Machine Learning", "Data Science"]
tags: ["topic modeling", "unsupervised learning"]
mathjax: true
---


# The concept of topic models
The most known topic model, which is the one that I used throughout my thesis as well, is called Latent Dirichlet Allocation (LDA). A topic model is an algorithm that describes an iterative process, whose goal is to uncover the latent themes ("topics" in the topic modeling jargon) that are assumed to generate the words of a document collection. In this post I present some basic concepts of topic modeling. 

The first question the above definition raises concerns these themes/topics: they are groups of words that tend to be semantically coherent and have different probabilities of appearance in text passages that discuss those topics. Imagine, for instance, a topic "Sports" and a topic "Chemistry". Words like "ball", "team" and "score" should have high probability of appearance when the topic is "Sports" and low probability when the topic is "Chemistry". On the other hand, words like "mercury", and "arsenic" should have high probability of appearance when the topic is "Chemistry". The tricky part here is that words do not belong to only a single topic, they belong to every topic, but their probabilities are quite different. For some topics a particular word may appear with very high probability while for other with very low.


Therefore, there is an inherent assumption in topic models that there is a set of topic underlying a text collection (=group of documents). Moreover, each document is a mixture of these topics. This means that a document $d_1$ can be seen as 

$$ d_i = 0.8\times\text{Sports} + 0.2\times\text{education}$$

which would mean that 80\% of the words of  $d_1$ come from the "Sports" topic, while 20\% from the "Education" topic. Once we have these mixture coefficients (0.8, 0.2 in the example), we can estimate a semantic similarity between documents or enable other types of applications. 


Of course, there is a catch in this. The problem is that we only observe the documents. Therefore, we need an algorithm for identifying (i) the topics (semantically coherent group of words)  and, (ii) which topics occur in a document and in what extent. The (i) point means we need to identify in some way that words like ["ball", "team"] and ["mercury", and "arsenic"] are semantically coherent. To do this we make use of the fact that words to tend to co-occur in similar contexts, should be similar (paraphrasing the expression "show me your friend and I will tell you who you are").  Once we achieve (i), (ii) can be seen as a by-product of it. 


$$\sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6} $$

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll's GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
