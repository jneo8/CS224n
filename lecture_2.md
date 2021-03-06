# Lecture 2

Word Vector Representations, word2vec

---

## How do we represent the meaning of a word?

Definition : meaning (Webster dictonary)

- the idea that is represented by a word, phrase, etc.
- the idea that a person wants to express by using words, signs, etc.
- the idea that is expressed in a word of writing, art, etc.

> meaning 的定義

Commonest linguistic way of thinking of meaning:

- signifier <-> signified(idea or thing) = denotation

> 最常見的語言思維方式
> 
> 語言中的單詞, 或者其他有意義的單元 <-> 用符號表示的概念 or 東西 = 詞的含義

---

## How do we usable meaning in a computer?

common answer : Use a taxonomy like WordNet that has hypernyms(is-a) relationships and synonym sets.

> Use nltk


---

## Problems with this discrete representation

- greart as a resource but missing nuances, e.g,
	- synonyms:
		- adept, expert, good, practiced, proficient, skillful

- Missing new words(impossible to keep up to date):
	wicked, badass, nifty, crack, ace, wizard, genius, ninja
	
- Subjective.
- Requires human labor to create and adapt.
- Hard to compute accurate word similarity.

> 困難點
> 
> * 同義詞
> 
> * 新詞
> 
> * 主觀
> 
> * 需要人力來創造
> 
> * 難以計算準確的文字相似度

---

## Problems with discrete representation

The vast majority of rul-based and stattistical NLP word regards words as atomic symbols: hotel, conference, walk


in vector space terms, this a vector with one 1 and a lot of zeroes 

[0, 0, 0, 0, 0, 0, 0, 0, 1]

Dimenisionality: 20k(speech)-50k(PTB)-500k(big vocab)-13M(google 1T) 

`one-hot` representation

it is a localist representatino

> `one-hot` representation
> 
> 用一個很長的向量來表示一個詞, 向量的長度為詞典的大小, 量只有一個 1, 其餘都為0
> 1 就表示這個  word, 在詞典中的位置
> It is a `locallist` representation.

---

## From symbolic to distributed representations

Its problem, e.g., for web search
	
- if user searches for `Dell notebook battery size`, we would like to match documents with `Dell laptop battery capacity`

- if user searches for `Seattle motel`, we would like to match documents containing `Seattle hotel`

But 
	motel [0, 0, 0, 0, 1]
	hotal [1, 0, 0, 0, 0]
	
**doesn't give any inherent notion of relationships between words, each word is notion to itself.**

> 這樣的 表示方法並無法 讓 word 之間有 概念上的關聯

---

## Distributional similarity based representsations

you can get a lot of value by representing a word by means of its neighbors

`you shall know a word by the company it keeps`

One of the most successful ideas of modern statistical NLP

- Government debt problem turning into `banking` crises as has happened in
- saying that Europe needs unified `banking` regulation to replace the hodgepodge

**These words will represent banking**

> 如果想要知道 banking 的意思, 在大量的文章中找尋臨近 banking 的字, 則這些字就可以代表 `banking` 的意思

---

## Word meaning is difined in terms of vectors

we will build a dense vector for each word type, chosen so that it is good at predicting other words appearing in its context.

**those other words also being represented by vectors... it all gets a bit recursive.**

```python
# 以向量表示詞
linguistics = [0.1, 0.7, 0.3, 0.1, 0.2, 0.4, 0.5]
```

> 詞在向量上的表示
> 
> 替每一種詞的型態 建構一個密集的向量, 以便他能夠預測上下文中所出現其他的詞.
> 
> **其他的詞也會以相同的型態被建構出來**

---

## Basic idea of learning neural network word embeddings


We define a model that aims to predict between a center word `Wt`, and context words in terms of word vectors 

```python
p(context | w) = ...
```

which has a loss function, e.g.,

```python
j = 1 - p(W-t | Wt)
```

we look at many positions t in a big language corpus

we keep adjusting the vector representations of words to minimize this loss

**Build a model predict a center word and words that appear in it's context**


> 建構一個 預測 word `Wt` 以及其上下文的  model
> 
> `W-t` mean context 中的所有word
> 
> `Wt` mean focus word
> 
> j 表示  loss function, 並且我們的目標是得到 j 的最小值

---

## Directly learning low-dimensional word vectors

Old idea. Relevant for  this lecture & deep learning:

- Learning representations by back-propagating errors (Rumelhart et al., 1986)
- **A neural probabilistic language model**(Bengio et al., 2003)
- NLP almost from Scratch(Collobert & Weston, 2008)
- A recent, even simpler and faster model: word2vec(Mikolov et al. 2013)

> 論文 history

---

## Main idea of word2vec

**Predict between every word and context words**

Two algorithms

- **Skip-grams(SG)**
	predict context words given target (position independent)

- Continuous Bag of Words(CBOW)
	Predict target word from bag-of-words context
	
Two (moderately efficient) training methods

- Hierarchical softmax
- Negative sampling

**Naive softmax**

> 講解 word2vec 用到的演算法 以及 訓練方法


---

## Skip-gram prediction

---

## Details of word2vec

For each word t = 1 ... T, predict surrounding words in a window of `radius` m of every word.

Objective function: Maximize the probability of any context word given the current center word: 

![j(θ)](https://github.com/jneo8/CS224n/blob/master/images/word2vec_1.png?raw=true)

Where θ represents all variables we will optimize

> 對每個單詞, 預測單詞半徑 m 中 的周圍單詞
> 
> Object function : 最大化當前中心詞的任何上下文單詞的概率
> 
> θ 表示可以被優化的 任何參數



## The objective function – details- Terminology: Loss function = cost function = objective function- Usual loss for probability distribution: Cross-entropy loss- With one-hot wt+j target, the only term left is the negative log probability of the true class- More on this later...



## Details of Word2Vec
predict surrounding words in a window of radius m of every word

For p(Wt+j | Wt') the simplest first formulation is

![p(0|c)](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_2.png?raw=true)

where o is the outside (or output) word index, c is thecenter word index

vc and uo are “center” and “outside”  vectors of indices c and o


`Softmax` using word c to obtain probability of word o

> 預測每個半徑 m 內 的周圍單詞, 
> 
> 最簡單的第一個公式
> 
> o 表示外部字 index, c 表示中心詞 index
> 
> vc, uo 分別表示 inside 和 outside 指標 c 的 向量
> 
> `Softmax` 使用 詞 c 來獲得 詞 o 的 概率



###  Dot product

![dot product](https://github.com/jneo8/CS224n/blob/master/images/word2vec_3.png?raw=true)

### Softmax function

![softmax function](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_4.png?raw=true)

### skipgram

![skipgram](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_5.png?raw=true)


---

##  To train the model: compute all vector gradients!

- We often define the set of all parameters in a model interms of one long vector θ

- In our case with d-dimensional vectors andV many words: 

![skipgram](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_6.png?raw=true)

- We then optimize these parameters

**Note: Every word has two vectors! make it simpler.**


> 優化參數 θ
> 
> 每個 w 都有兩個向量


---

## Derivations of gradient

> 接下去會花很多時間推導公式

- whiteboard - see video if you are not in class
- The basic Lego piece
- Useful basice 

![Useful basice ](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_7.png?raw=true)

- if in doubt: write out with indices
- Chain rule! If y = f(u) and u = g(x), i.e. y=f(g(x)), then 

![Chain rule](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_8.png?raw=true)

- Chain rule! If y = f(u) and u = g(x), i.e. y = f(g(x)), then:


![Chain rule](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_9.png?raw=true)

![Chain rule](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_10.png?raw=true)

---

## Interactive Whiteboard Session!

![j(θ)](https://github.com/jneo8/CS224n/blob/master/images/word2vec_1.png?raw=true)


Let’s derive gradient for center word togetherFor one example window and one example outside word:

![11](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_11.png?raw=true)

You then also also need the gradient for context words (it’s similar; left for homework). That’s all of the paramets θ here.

![12](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_12.png?raw=true)

![13](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_13.png?raw=true)

![14](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_14.png?raw=true)

![15](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_15.png?raw=true)


---

## Calculating all gradients!
- We went through gradient for each center vector v in a window- We also need gradients for outside vectors u- Derive at home!- Generally in each window we will compute updates for all parameters that are being used in that window.- For example, window size m = 1, sentence: “We like learning a lot”- First window computes gradients for:- internal vector vlike and external vectors uWe and ulearning- Next window in that sentence?


---

##  Cost/Objective functions

We will optimize (maximize or minimize) our objective/cost functionsFor now: minimizeàgradient descentTrivial example: (from Wikipedia)Find a local minimum of the functionf(x) = x4−3x3+2, with derivative f'(x) = 4x3−9x2

![16](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_16.png?raw=true)

```python
x_old = 0
x_new = 6
precision = 0.000001
def f_derivative(x):
	return 4 * x**3 - 9 * x**2

while abs(x_new - x_old) > precision:
	x_old = x_new
	x_new = x_old - eps * f_derivative(x_old)
	
print("Local minimun occurs at", x_new)

# Substacting a fraction of the gradient moves you towards the minimum!

```
---

## Gradient Descent

- To minimize   over the full batch (the entire training data) would require us to compute gradients for all windows

- Updates would be for each element of θ :

![17](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_17.png?raw=true)

- With step size α- In matrix notation for all parameters:

![18](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_18.png?raw=true)

---

## Vanilla Gradient Descent Code

![19](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_19.png?raw=true)

```python 
while True:

	theta_grad = evaluate_gradient(J, corpus, theta)
	theta = theta - alpha * theta_grad
```

---


## Intuition

For a simple convex function over two parameters. 

Contour lines show levels of objective function


![20](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_20.png?raw=true)

---

## Stochastic Gradient Descent

But Corpus may have 40B tokens and windows
- You would wait a very long time before making a single update!- Very bad idea for pretty much all neural nets!- Instead: We will update parameters after each window t àStochastic gradient descent (SGD)

![21](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_21.png?raw=true)


```python
while True:
	window = sample_window(corpus)
	theta_grad = evaluate_gradient(J, window, theta)
	theta = theta - alpha * theta_grad
```






