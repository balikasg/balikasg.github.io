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

To illustrate this better in the following figure I illustrate the topics that LDA has discovered 

<!--  -->
![LDA inference result]({{ site.url }}/assets/example.png "LDA: topics discovered for the words of the excerpt.)


