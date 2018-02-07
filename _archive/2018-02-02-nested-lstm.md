---
layout: post
title: "Coping with Hierarchical Time Dependency, Nested LSTM"
description: "Understanding NLSTM, and the intuition behind it."
tags: [metalearning, machinelearning]
---

Let the superscript be the memory depth of 


{% raw %}
$$
\begin{align*}
h^1_t, c^1_t &= \text{NLSTM}(c^1_{t-1}, c^2_{t-1}, ..., c^D_{t-1})(h^1_{t-1}, x^1_t) \\
c^1_t, c^2_t &= \text{NLSTM}(c^2_{t-1}, ..., c^D_{t-1})(h^2_{t-1}, x^2_t)\\
&\vdots\\
c^{d-1}_t, c^d_t &= \text{NLSTM}(c^d_{t-1}, ..., c^D_{t-1})(h^d_{t-1}, x^d_t)\\
&\vdots\\
c^{D-1}_t, c^D_t &= \text{LSTM}(c^D_{t-1})(h^D_{t-1}, x^D_t) 
\end{align*}
$$
{% endraw %}


{% raw %}
$$
\begin{align*}
i^d_t &= \sigma(x^d_t W^d_{xi} + h^d_{t-1} W^d_{hi} + b^d_i)\\
f^d_t &= \sigma(x^d_t W^d_{xf} + h^d_{t-1} W^d_{hf} + b^d_f)\\
g^d_t &= \sigma(x^d_t W^d_{xg} + h^d_{t-1} W^d_{hg} + b^d_g) \\
h^{d+1}_{t-1} &= i^d_t \odot g^d_t\\ 
x^{d+1}_t &= f^d_t \odot c^d_{t-1}\\
c^d_t &= \text{NLSTM}(c^{d+1}_{t-1}, ...,c^D_{t-1})(h^{d+1}_{t-1}, x^{d+1}_t)\\
o^d_t &= \sigma(x_t W_{xo} + h_{t-1} W_{ho} + b_o)\\
h^d_t &= o^d_t \odot \sigma(c^d_t) 
\end{align*}
$$
{% endraw %}