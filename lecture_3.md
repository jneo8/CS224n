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


---

## Example : window based co-occurrence matrix

- window length (more common: 5 - 10)
- symmetric(irrelevant whether left or right context)
- Example corpus 
	- i like deep learning
	- i like NLP
	- i enjoy flying
	
![](https://github.com/jneo8/CS224n/blob/master/images/glove_10.png?raw=true)

> 圖表 是  window based co-occurrence matrix, 
> 
> 以第一行來說, 表示了  word `i` 在長度唯一的狀況下, 臨近詞的統計

<br>

## Problem with simple co-occurence vectors

Increase in size with vocabulary

Very high dimensional: require a lot of storage

Subsequent classification models have sparsity issues

-> `Models are less rebost`

> 隨著詞彙增加, 會需要大量儲存, 效能不佳
> 
> 分類器模型會有稀疏性問題

---

## Solution : Low dimensional vectors

- idea: store "most" of the important information in a fixed, small number of dimensions: a dense vector

- usually 25 - 100 dimensions, similar to word2vec

- How to reduce the dimensionality?

> 為了提高效能, 使用了跟 word2vec 相同的概念, 只存了重要的 dimensions, 大多是 25 - 100 個, 
> 但是問題是要如何縮減  dimensions


---

## Method 1 : dimensionality Reduction on x

singular Value Decomposition of co-occurrence matrix `X`

![](https://github.com/jneo8/CS224n/blob/master/images/glove_11.png?raw=true)

> 使用 SVD 奇異值分解 co-occurrence martix, 可以有效降低矩陣大小


<br>

## SVD python example

```python

"""Sample code SVD."""
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os


def main():
    la = np.linalg

    words = ['i', 'like', 'enjoy', 'deep', 'learning', 'NLP', 'flying', '.']
    x = np.array([
        [0, 2, 1, 0, 0, 0, 0, 0],
        [2, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 0],
    ])

    u, s, vh = la.svd(x, full_matrices=False)

    for i in range(len(words)):
        plt.text(u[i, 0], u[i, 1], words[i])

    plt.savefig('svd')
    subprocess.call(['catimg', '-f', 'svd.png'])
    os.remove('svd.png')
    
 if __name__ == '__main__':
    main()
```

![](https://github.com/jneo8/CS224n/blob/master/images/glove_12.png?raw=true)


---

## Hacks to x

- problem: function words (the, he, has) are too frequent -> syntax has too impact. Some fixes:
	- min(x, t), with t~100
	- lgnore them all

- Ramped windows that count closer word more
- Use pearson correlations instead of counts, then set negative values to 0
- +++

> Hacks 方式
> 
> 給太頻繁出現的字, 例如 stopword, 最大值 or 忽略他們
> 
> windows 針對較近的字計算更多次
> 
> 使用 `pearson correlations` 而不是 count, 將負值設為 0


---

## Interesting semantic patters emerge in the vectors

![](https://github.com/jneo8/CS224n/blob/master/images/glove_13.png?raw=true)

![](https://github.com/jneo8/CS224n/blob/master/images/glove_14.png?raw=true)

![](https://github.com/jneo8/CS224n/blob/master/images/glove_15.png?raw=true)


An Improved Model of SemanRc Similarity Based on Lexical Co-Occurrence Rohde et al. 2005


> 效果展示

---

## Problem with SVD

Computational cost scales quadratically for n x m matrix:

O(mn2) flops (when n < m)
-> Bad for millions of words or documents

Hard to incorporate new words or documents
Different learning regime than other dl models


> n * m 矩陣的計算成本
> 很難納入新詞, 相較於其他 dl models, 因為一旦加入新詞必須重跑整個 流程

---

## Count based vs direct prediction

- Count based
	- LSA, HAL(lund & Burgess)
	- COALS, Hellinger-PCA (Rohde et al , lebret & Collobert)

	- advantage	
		- Fast training
		- Efficient usage of statistics

	- disadvantage
		- Primarily udes to capture word similarity
		- Disproportionate importance given to large counts

- Direct prediction
	- Skip-gram/CBOW (mikolov et all)
	- NNLM, HLBL, RNN (Bengio et al; Collobert & Weston;Huang et al; Mnih & Hinton; Mikolov et al; Mnih & Kavukcuoglu)

	
	- advantage
		- Scale with corpus size
		- inefficient usage of statistics

	- disadvantage
		- Generate improved performance on other tasks
		- Can capture complex patterns beyond word similarity

> 比較了 Count based & Direct prediction 的優缺點


---

Combining the best of both worlds : GloVe

![](https://github.com/jneo8/CS224n/blob/master/images/glove_16.png?raw=true)

- Fast training
- Scalable to huge corpora
- Good performance even with small corpus, and small vectors

> Glove 
> 
> 訓練快
> 
> 可以適用於大型 or 小型語料庫, 都有良好的表現

---

## What to do with the two sets of vectors?

- We end up with U and V from all the vectors u and v (in columns)
- Both capture similar co-occurrence information. It turns out, the best solution is to simply sum them up:

![](https://github.com/jneo8/CS224n/blob/master/images/glove_17.png?raw=true)

- One of many hyperparameters explored in `Glove`: Global Vectors for word Representation, Pennington et al. (2014)

---
	
## Glove results 

Nerest words to `frog`

- frogs
- toad
- litoria
- leptodactylidae
- rana
- lizard
- eleutherodactylus

> Glove 效果, frog 的相近詞

---

## How to evaluate word vectors

- Related to general evaluation in NLP; intrinsic vs extrinsic
- Intrinsic
	- Evaluation on a specific/intermediate subtask
	- Fast to compute
	- Helps to inderstand that system
	- Not clear if really helpful unless correlation to real task is established

- Extrinsic
	- Evaluation on a real task
	- Can take a long time to compute accuracy
	- Unclear if the subsystem is the problem or its interaction or other subsystem 
	- if replacing exactly one subsystem with another improves accuracy -> winning!

> 如何評估  word vectors?

---

## Intrinsic word vector evaluation

- word vector angalogies
	`man:woman :: king:?`
	
	![](https://github.com/jneo8/CS224n/blob/master/images/glove_18.png?raw=true)
	
	![](https://github.com/jneo8/CS224n/blob/master/images/glove_19.png?raw=true)
	
- Evalute word vectors by how well their cosine distance after addition captures intuitive senmantic and syntactic analogy questions
- Discarding the input words from the search!
- Problem: What if the information is there but not linear?

> 透過加法的餘弦距離來評估 word vector
> 或者放棄輸入字
> 
> 問題是 如果不是線性的？

![](https://github.com/jneo8/CS224n/blob/master/images/glove_20.png?raw=true)
![](https://github.com/jneo8/CS224n/blob/master/images/glove_21.png?raw=true)
![](https://github.com/jneo8/CS224n/blob/master/images/glove_22.png?raw=true)


<br>

## Details of intrinsic word vector evalution

- Word Vector Analogies: SyntacRc and Seman(c examples from 
	[hmp://code.google.com/p/word2vec/source/browse/trunk/quesRons- words.txt](hmp://code.google.com/p/word2vec/source/browse/trunk/quesRons- words.txt)
	
--- 

## Analogy evaluation and hyperparameters

- very careful analysis: Glove word vectors

![](https://github.com/jneo8/CS224n/blob/master/images/glove_23.png?raw=true)


> dimension 數量大的不一定比較準, 但是 size 大的會越準
> 在利用 餘弦距離取得 相對詞  Glove word vectors 表現良好(最下面一行)

- Asymmetric context (only words to the lel) are not as good

![](https://github.com/jneo8/CS224n/blob/master/images/glove_24.png?raw=true)

- Best dimensions ~300, slight drop-off afterwards
- But this might be different for downstream tasks
- Window size of 8 around each center word is good for glove vectors

> 評估 hyperparameters 的影響

![](https://github.com/jneo8/CS224n/blob/master/images/glove_25.png?raw=true)


![](https://github.com/jneo8/CS224n/blob/master/images/glove_26.png?raw=true)

## Another intrinsic word vector evaluation

- word vector distances and their correlation with human judgments
- Example dataset: WordSim353

[hmp://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/](hmp://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)
  
  
---









  











	

