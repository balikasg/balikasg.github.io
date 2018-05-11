---
title:  "Why would you organize a meet-up?"
categories: ["Machine learning", "Information retrieval", "Word embeddings"]
tags: ["data science"]
mathjax: true
---

In this post I would like to detail the ideas behind our ECIR 2018 paper, titled "Cross-lingual Document Retrieval using
Regularized Wasserstein Distance". While the paper's title may sound complicated, I believe the central idea is straight-forward. The paper builds on the seminal work of [M. Kusner et al.](http://proceedings.mlr.press/v37/kusnerb15.pdf) where they propose to combine the expressiveness of word embeddings with the theory of optimal transport. 

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>


# The problem of optimal transport
The problem of optimal transport is a well-studied problem with a long history starting as back as 1940.  In its original formulation by Monge the problem is defined as follows: Given two distributions of equal masses, find a transport map $\gamma$  which transfers the first distribution to the second and minimizes the associated transport cost. There are mathematical formulations for the problem both for the continuous and the discrete case. For simplicity, I will skip the mathematical forumaltion of the problem as it can be found both in the paper and in several textbooks: e.g., from G. Peyré and M. Cuturi [here](https://optimaltransport.github.io/book/) and from L. Ambrosio [here](http://cvgmt.sns.it/media/doc/paper/1008/trasporto.pdf). 

To makes things more clear, let's graphically illustrate the problem using a bipartite graph as an example. Imagine the problem of having two factories preparing croissants, factory $S_1$ and $S_2$ and, three bakeries that sell these croissants: $T_1$, $T_2$ and $T_3$. Transfeting croissants from the sources $S_i$ to the targets $T_j$ involves different costs depending on their distance. The optimal transport problem seeks the solution of transfering all the croissants from the factories $S_i$ to the bakeries $T_j$ minimizing the associated cost of transfer. In the following figure for instance, I illustrate the source and the target points, the availability (100+200) and the demand (100+100+100) for croissants, and the costs from $S_1$ to $T_j$.

![Optimal transport example]({{ site.url }}/assets/optimalTransportExample.pdf)



