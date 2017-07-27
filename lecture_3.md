# GloVe Vectors for word Representation


---

## Review : Main idea of word2vec

- go through each word of the whole corpus
- Predict surrounding words of each (window's center)
	word
	
![](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_2.png?raw=true)

	
- Then take gradients at each such windows for SGD

> 複習 word2vec 的重點

---

## Stochactic gradients with word vectors!

- But in each window, we only have at most 2m + 1 words, is very sparse!

![](https://github.com/jneo8/CS224n/blob/master/images/glove_1.png?raw=true)

- We may as well only update the word vectors that actually appear!
- Solution: 
	either you need sparse matrix update operations to only update certain columns of full enbedding matrices `U` and `V`, or you need to keep around as hash for word vectors


![](https://github.com/jneo8/CS224n/blob/master/images/glove_2.png?raw=true)


- If you have millions of word vectors and do distributed cumputing, it is important not have to send gigantic updates around!


> 每個 window 中, 只有 2m + 1個字
> 
> 實際上可以只更新 實際出現的 word vector
> 
> 如果我們大量的  word vector, 重要的是 不要做 巨大的更新

---

## Approximations: Assignment 1

- The normalization factor is too computationally expensive.


![](https://github.com/jneo8/CS224n/blob/master/images/wod2vec_2.png?raw=true)


- Hence, in Assignment 1, you will implement the skip-gram model with `negative sampling`

- Main idea: train binary logistic regressions for a true pair(center word and word in its context window) versus a couple of noise pairs (the center word paired with random word)

> `normalization factor`(也就是公式下面的部分), 因為需要 sum 所有, 因此太貴了
> 
> 我們需要使用  `negative sampling` 來實現 `skip-gram model`
> 
> Mean idea : 
> > 對 windows 內部的 word 以及 兩個  noise pairs (center word 搭配上 random word) 訓練  `binary logistic regressions`


---

## Ass 1 : The skip-gram model and negative sampling

- From papar : "Distributed Respresentations of words and Phrases and their Compositionality " (Mikolov et al. 2013)- Overall objective function: 

![](https://github.com/jneo8/CS224n/blob/master/images/glove_3.png?raw=true)


![](https://github.com/jneo8/CS224n/blob/master/images/glove_4.png?raw=true)

- The sigmoid function! 
![](https://github.com/jneo8/CS224n/blob/master/images/glove_6.png?raw=true)
We will become good friends soon

- So we maximize the probability of two words co-occurring in first log 

 ![](https://github.com/jneo8/CS224n/blob/master/images/glove_5.png?raw=true)

- Slightly clearer notation : 
  ![](https://github.com/jneo8/CS224n/blob/master/images/glove_7.png?raw=true)
  
 - We take k negative samples
 - Maimize probability that real outside word appears, minimize prob. that random words appear around center word
 
 - ![](https://github.com/jneo8/CS224n/blob/master/images/glove_8.png?raw=true)
  	The unigram distribution U(w) raised to the 3/4 power (we provide this function in the starter code).
  	
  - The power makes less frequent words be sampled more often

> 這邊解釋了 `skip -gram` 運作的方式
> 
> 使用  [sigmoid function](https://zh.wikipedia.org/wiki/S函数)
> 
> 最大化了 first log(公式前半段) 共同出現兩個 word 的 probability
> 
> 取 k 個負樣本,
> 
> 使用 U(w)3/4z 使得 出現頻率較少的詞被更頻繁抽樣


---

## Ass 1 The continuous bag of words model

- Main idea for continuous of words (CBOW) : predict center word wrom sum of surrounding word vectors instead of predict surrounding single words from center word as in skip-gram model.

- To make assiment slightly easier:
	
	Implementation of the CBOW model is not required(you can do it for a couple og bonus points), but you do have to the written problem on CBOW


> CBOW 的 主要思想是預測中心詞和周圍詞向量的 和

> CBOW model 並不一定要執行, 詳見上

---

## Word2vec improves object function by putting similar words nearby in space

![](https://github.com/jneo8/CS224n/blob/master/images/glove_9.png?raw=true)

---

## Summary of word2vec

- Go through each word of the whole corpus
- Predict surrounding words of each word
- This captures cooccurrence of words one at a time
- why not capture cooccurence counts directly

**Yes we can**

With a co-occurrence matrix X

- 2 options: full document vs windows
- Word-document co-occurrence matrix will give general topics (all sports terms will have similar entries) leading to "Latent Semantic Analysis"


- Instead: Similar to word2vec, use window around each word -> captures both syntactic(POS) and semantic information.



  











	

