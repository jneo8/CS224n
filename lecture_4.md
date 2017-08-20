# Overview today

- Classification background
- updating word vectors for classification
- Window classification & cross entropy error derivation tips
- A single layer neural network
- Max-Margin loss and backprop


---

# Details of softmax

![](https://github.com/jneo8/CS224n/blob/master/images/l4_1.png?raw=true)

[wiki](https://zh.wikipedia.org/wiki/Softmax函数)

##  Cross entropy error

> 假設 所有的 target 只有 1 & 0
> 
> p = [0, 1, 0, 1, 0, 0]
> 
> q 為 computed probability 


![](https://github.com/jneo8/CS224n/blob/master/images/l4_2.png?raw=true)


**Because of one-hot p, the only term le= is the nega6ve log probability of the true class**


##  Classification : Regularization

- Really full loss function over any dataset includes `regularization` over all parameters `0`:

![](https://github.com/jneo8/CS224n/blob/master/images/l4_3.png?raw=true)

##  Window classifica6on

- idea: classify a word in its context window of neighboring words.

##  Updating concatenated word vectors

- What is the dimensionality of the wondow vector gradient?

![](https://github.com/jneo8/CS224n/blob/master/images/l4_4.png?raw=true)

- x is the window, 5 d-dimensional word vectors, so the derivative wrt to x has to have the same dimensionality:


![](https://github.com/jneo8/CS224n/blob/master/images/l4_5.png?raw=true)


##  A note on matrix implementations

- Looping over word vectors instead of contatenating them all into one large martix and then multiplying the softmax weights with that matrix

```python
from numpy import random
n = 500
d = 300
c = 5
w = random,rad(C, d)

wordvectors_list = [random.rand(d, 1) for i in range(n)]
wordvectors_one_matrix = random.rand(d, n)

%timeit [w.dot(wordvectors_list[i] for i in range(n)]
%timeit w.dot(wordvectors_one_matrix)
```

##  Softmax (= logistic regression) alone not very powerful

- softmax only gives linear desision bound daries in the original space

- With little data that can be a good regularizer
- with more data it is very limiting


---


# From logistic regression to neural nets

**A neural network = running several logistic regressions at the same time**

If we feed ad vector of input through a bunch of logistic regression functions, then e get a vector of outputs ...

But we don't have to decide ahead of time what variables these logistic regressions are tring to predict!


## Summary: Feed-forward Computa6onComputing a window’s score with a 3-layer neural net: s = score(museums in Paris are amazing )


![](https://github.com/jneo8/CS224n/blob/master/images/l4_6.png?raw=true)


---


# The max-margin loss














