---
layout: post
title: "How to make sense of cross entropy"
description: "Understanding the most used loss function in deep learning - cross entropy."
tags: [loss function, machinelearning]
---

Cross entropy is the most widely used loss function in deep learning applications. Having a firm grasp on this final layer of model training will help you understand what is actually going on under the hood. 

I am going to make this dead simple. We can view cross entropy in two ways. The probabilistic way and the information theoretic way. In my opinion, the probabilistic way makes more intuitive sense when applied to machine learning, so I will introduce it first. For the inquistive mind, you can read on to the second part about the information theory.


### Probabilistic View
Cross entropy is defined in a discrete case for one training example as

{% raw %}
$$\text{H}(p(y), q(y)) = \sum_{y \in \Omega} -p(y) \log q(y) = -\text{E}_{p}[\log q(\hat{Y}=y)]$$
{% endraw %}

, where \\(q\\) is the estimated probability distribution of the output label, \\(p\\) is the actual distribution of the output label, \\(\Omega\\) is the space of all the possible output label. The cross entropy is just the minus of the expected log likelyhood of your estimation. When you minimize the cross entropy loss during training, you are essentially just maximizing (due to the minus sign) the expected log likelyhood of your estimation. In other words, this is no different than the maximum log likelyhood estimation in most of the regression and classification that you are familiar with in statistics.

Let us write this down in a more general, multiple data point case. Assuming you have \\(n\\) training data points, and \\(m\\) output labels,

{% raw %}
$$ \text{loss} = \sum_{i=1}^n\text{H}(p(Y_i), q(\hat{Y}_i)) = \sum_{i=1}^n\sum_{j=1}^m -p(Y_i = y_j) \log q(\hat{Y}_i = y_j)$$.
{% endraw %}

Note how I seperated the random variables, \\(Y_i\\) and \\(\hat{Y}_i\\), and the values, \\(y_j\\), to make the semantics clearer<sup>[1](#footnote1)</sup>.

For a 2 classes problem, as in logistic regression, some people like to express cross entropy in this alternative form

{% raw %}
$$ \text{loss} = \sum_{i=1}^n \eta_i log (q_i) + (1 - \eta_i) log(1 - q_i)$$.
{% endraw %}

, where \\(\eta_i\\) is the \\(i\\)th output label in the two cases\\(\\left\\{0, 1\\right\\}\\) and \\(q_i = q(\hat{Y}_i = 1)\\). You will see this form in some machine learning textbook.


### Information Theoretic View
Cross entropy can also be seen as the quantity of information you are expected to receive given an alternitive decoding scheme. What do we mean by alternitive decoding scheme here? To understand this, we need to first understand what information means.

In information theory, the quantity of information for a receiver is related to the probability distribution of the signal sending out from the source. More precisely, the information content is quantified by \\(-\\log\,p(E)\\) of some event \\(E\\). For example, to encode a fair coin toss, which has 0.5 probability of having a head. You recieve \\(-\\log(0.5) = 1\\) bit of information each time you see a head shows up on the receiver end. We call this bit one shannon. Note that the information content does not directly correspond to how many digits in order to encode this message; the informtion content here correspond to how rare the event actually happens. The consequence of how this rareness influences encoding is a much more complex issue, it demands a solid understanding of encoding concept (Huffman encoding, data compression, ..., etc) and the real transmission scheme used by engineers; I listed some good resources at the bottom to help you understand it. For now, just bare in mind that rareness equals information content.

This makes intuitive sense. Let's say if you have a coin that has head on both sides. There is no need to check whether it's head or tail. Therefore, the information content is \\(-\\log 1 = 0\\) shannon. On the contrary, if there is a rare event, such as an earthquak that have 1/4096 chance of happening tomorrow; it is worth 12 shannon. Knowing a rare event is much more informative and valuable than knowing the mundane.

Over the course of transmission, the average information content, or shannon, the receiver expect to see is known as Shannon entropy, and expressed as

{% raw %}
$$ \text{H}(p) = \sum_{E \in \mathcal{E}} -p(E)log\, p(E) $$.
{% endraw %}

This tells you how much information content you are expect to receive, given perfect knowledge of the underlying probability distribution. This looks very similar to cross entropy which reads

{% raw %}
$$ \text{H}(p, q) = \sum_{E \in \mathcal{E}} -p(E)log\, q(E) $$.
{% endraw %}

 The only difference is in the log term. The \\(q\\) here correspond to the probability distribution of events that the receiver thinks it is. Let's say if a receiver have perfect knowledge of what the sending end probability distribution looks like, then the cross entropy will just be equal to entropy. What if the receiver have a wrong decoding scheme (or wrong assumption about the probability distribution on the sender end)? It turns out that you will always be too optimistic about how much information you will receive. In other words,

{% raw %}
$$ \text{H}(p, q) \ge \text{H}(p) $$.
{% endraw %}

When you are using a wrong decoding scheme, you will always expect more information to be received than you actually do receive. Having wrong expectation will always result in dissapointment (surprise surprise!). You can calculate the difference between your expectation and the true expectation. This is called Kullback–Leibler divergence and defined as

<!-- {% raw %}
$$ D_{KL}(p) = \text{H}(p, q) \le \text{H}(p) $$.
{% endraw %} -->

{% raw %}
$$ D_{KL}(p\|q) = \text{H}(p, q) - \text{H}(p) \ge 0 $$.
{% endraw %}

KL divergence is the one that makes more intuitive sense when used as a loss function since it measures how different \\(p\\) and \\(q\\) are, and we are always onto the quest of approximating \\(p\\) by \\(q\\). In fact, KL divergence is very often used in machine learning as a loss function; for instance, when applied to variational type of problem. Since the underlying distribution \\(p\\) does not change, the entropy term is a constant; optimizing KL divergence is just the same as optimizing cross entropy.

Phewww, that is a lot to absorb. Why should we learn both view points? As Richard Feynman [pointed out](https://www.youtube.com/watch?v=NM-zWTU7X-k&t=2s), if there are two competing theories that are mathematically equivalent, it is still useful to learn both, since in some cases only one of those theories will provide you the right kind of mental understanding to unlock deeper truth. So, just like any ML practitioner, I chose to suck it.

There is still a mystery in the deep learning field why cross entropy loss works much better in training a deep network than using MSE. The rule of thumb nowadays is you need to discretize your output label even if you are faced with a continuous output space. I am planning on looking into this issue in the future because I constantly encounter regression type of problem in my daily work. I welcome any suggestion and discussion in this matter.

<small>
<a name="footnote1">1</a>: Note in the literature, people often write \\(p(Y=y)\\) as \\(p(y)\\), and \\(Y\\) sometimes not subscripted. This shorthand often causes trouble in understanding the operation that is actually being done. For example, in the regular model fitting when we deal with n output label, the outputs are actually modeled as n random variables \\({Y_1,...,Y_n}\\) and each random variable has its own probability distribution. The estimations are another n random variables \\({\hat{Y}_1,..., \hat{Y}_n}\\). Using just \\(Y\\) without subscript gave you the wrong mental picture that all output labels share the same distribution, while in fact they don't.
</small>

Some useful links.

1. [A Mathematical Theory of Communication](http://worrydream.com/refs/Shannon%20-%20A%20Mathematical%20Theory%20of%20Communication.pdf) - the seminal work of Claude Shannon
2. [Understanding Shannon's Entropy metric for Information](https://arxiv.org/pdf/1405.2061.pdf)
3. [Wiki - Cross Entropy](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjJqcS3y6fTAhUosVQKHWu_B98QFgglMAA&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FCross_entropy&usg=AFQjCNFxTG-ICjvkx598wpX_6xhnq1Tw3Q&sig2=ZR6GFDjnUveOqbCLS_y1BQ)
4. [A Friendly Introduction to Cross-Entropy Loss](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0ahUKEwjJqcS3y6fTAhUosVQKHWu_B98QFgg8MAI&url=https%3A%2F%2Frdipietro.github.io%2Ffriendly-intro-to-cross-entropy-loss%2F&usg=AFQjCNEJfYOWMzbYXcmLYJ2-iWFBB-Vj1Q&sig2=pD32vI_znICFohO-nXXchg)