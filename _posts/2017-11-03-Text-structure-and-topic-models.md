---
title:  "Text structure and topic models"
categories: ["Machine Learning", "Data Science"]
tags: ["topic modeling", "unsupervised learning"]
mathjax: true
---


In the previous post I introduced Latent Dirichlet Allocation, which is a popular topic model. In this post I highlight one of the limitations of LDA: its inability to model text structure. After demonstrating the reasons why, I am introducing part of my published work that alleviates this limitation.


<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>
 

# Topic models and the exhangeability assumption
Topic models like Latent Dirichlet Allocation (LDA) are a class of unsupervised models used to uncover the hidden themes that have generated a text collection. An explicit assumption underlying LDA is exchangeability that stipulates that the topics of the words are conditionally independent given the document topic distributions. In other words, documents are represented as bag-of-words and the association of topics with words does not account for any type of data structure. This is particularly important as it enables fast inference. Since, however, text structure is ignored it can lead to  suboptimal results. 

To illustrate this better in the following figure I show the topics that LDA has discovered is a short excerpt from a Wikipedia document for a movie. There are two main observations from the figure: 

+ LDA successfully manages to associate most of the document words with the `Cinema` topic. Since the document concerns a movie this is a positive point for the model.

+ There are, however, several short text passages like `film noir classic`, `Brian Donlevy` and `Richard Widmark` whose component words have been associated with different topics. Such short segments for humans are topically consistent, and would hope that LDA could be able to model that. 


![LDA inference result]({{ site.url }}/assets/example.png)


The limitation of LDA that the previous figure exposed is due to the exchangeability assumption and the the bag-of-words representation of the documents of a collection. To make this more clear, here is the generative story of LDA. Observe how every notion of structure  is absent:

* For $k \in [1, K]: \phi_k \sim Dir(\beta)$
* for each document $d_i$ :
   * sample topic distribution $\theta_i \sim Dir(\alpha)$ 
   * for each word $w_{i,j}, j \in [1, N_i ]$ :
      * sample topic $z_{i,j} \sim Cat(\theta_i) $
      * sample word $ w_{i,j} \sim Cat(\phi_{z_{i,j}} $



Here, $Dir$ is a draw from the Dirichlet distribution, $\alpha, \beta$ are the Dirichlet concentration parameters and $Cat$ stands for a draw from a Categorical distribution. Also, $\alpha, \beta$ being scalars implies symmetric priors; these values are usually set such that $\alpha, \beta \lt\lt 1$ so as to obtain sparse word and document topic distributions.
In this generative story, that is a set of assumptions on how a document collection is generated,  notice how sampling the topics of the words from the document topic distribution $\theta_i$ is a sequential process that ignores structure. Hence, the behavior of LDA where structure is not taken into account as we detailed in the example figure is due to the model's design. 

LDA is structure-agnostic in the sense that it cannot assign consistently topics to small segments by design. Since this may be counter-intuitive for humans, it would be nice if one could extend LDA to account for such text structure. Knowledge of text structure in the form of short text spans consisting of contiguous words has been shown to be beneficial for machine learning and natural language processing tasks. Several well studied approaches like linguistic analysis of sentences or heuristics like $n$-grams are heavily used in text mining applications like classification, clustering and named entity recognition.

To overcome this limitation of LDA, we have recently proposed two extensions of the model. The first assumes a complete dependence between the topics of the words of a segment. To do that it modifies the generative story of the topic model so that the topic of the segment words is sampled once. In this case, by definition, the words of a segment are topically coherent and the dependence is maximal. This work is described in [1]. The second proposes a more flexible dependence mechanism that relies on copulas. [Copulas](https://en.wikipedia.org/wiki/Copula_(probability_theory) "Wikipedia article for copulas") are a powerful statistical framework that allow to model joint cumulative distribution function as a function of univariate marginal functions. The nice thing here is that one can model the dependence between two (or more) random variables irrespective of the marginal distributions of these random variables.  Copulas are more flexible as they allow different degrees of dependence, which results in better performance. This work is discussed in detail in [2]. 

In the following two posts I will describe this two contributions in more detail! 

<!--

![Copulas: sample from a Frank copula]({{ site.url }}/assets/copulas.gif)
--> 

[1]: G. Balikas, MR Amini, M. Clausel: Latent Dirichlet Allocation, SIGIR, 2016  [[pdf]](https://arxiv.org/abs/1606.00253 "SIGIR'16 paper")

[2]: G. Balikas, H. Amoualian, M. Clausel, E. Gaussier, MR Amini: Modeling topic dependencies in semantically coherent text spans with copulas, COLING, 2016 [[pdf]](http://aclweb.org/anthology/C16-1166 "COLING'16 paper")



