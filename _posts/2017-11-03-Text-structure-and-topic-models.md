---
title:  "Text structure and topic models"
categories: ["Machine Learning", "Data Science"]
tags: ["topic modeling", "unsupervised learning"]
mathjax: true
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>


# Topic models and the exhangeability assumtpion

Topic models like Latent Dirichlet Allocation (LDA) are a class of unsupervised models used to uncover the hidden themes that have generated a text collection. An explicit assumption underlying LDA is exchangeability that stipulates that the topics of the words are conditionally independent given the document topic distributions. In other words, documents are represented as bag-of-words and the association of topics with words does not account for any type of data structure. This is particularly important as it enables fast inference. Since, however, text structure is ignored it can lead to  suboptimal results. 

To illustrate this better in the following figure I show the topics that LDA has discovered is a short excerpt from a Wikipedia document for a movie. There are two main observations from the figure: 

+ LDA successfully manages to associate most of the document words with the `Cinema` topic. Since the document concerns a movie this is a positive point for the model.

+ There are, however, several short text passages like `film noir classic`, `Brian Donlevy` and `Richard Widmark` whose component words have been associated with different topics. Such short segments for humans are topically consistent, and would hope that LDA could be able to model that. 


![LDA inference result]({{ site.url }}/assets/example.png)


The limitation of LDA 


