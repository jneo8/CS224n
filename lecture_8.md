# Recurrent Neural Network


## Language Models 

A language models computers a probability for a sequence of words : P(W1, .....Wt)


- Useful for machine translation

    - Word ordering:
        p(the cat is small) > p(small is the cat)


    - Word choice
        p(walking home after school) > p(walking house after school)

---

## Traditional Language Models

- Probability is usually conditioned on window of n previous words

- An incorrect but necessary Markov assumption!(馬可夫假設)

![](https://github.com/jneo8/CS224n/blob/master/images/l8_1.png?raw=true)

- To estimate probabilites, compute for unigrams and bigrams

![](https://github.com/jneo8/CS224n/blob/master/images/l8_2.png?raw=true)

- Performance improves with keeping around higher n-grams counts and doing smoothing and so-called backoff(if 4-gram not found, try 3-gram, etc)

- There are *A LOT* of n-grams! -> Gigantic RAM requirements!

- Recent state of the art: Scalable Modified Kneser-Ney Language Model Estimation by Heafield et al.: 

	**Using one machine with 140 GB RAM for 2.8 days, we built an unpruned model on 126 billion tokens**
	
	
> 主要講說傳統的model, 建構在不正確但是又是必須的馬可夫假設
> 效能建構在大量的  n-grams 上
> > 又做了所謂的平滑, 比如說  4-gram 找不到換  3-gram
> > 
> > 會消耗大量的RAM


---

# Recurrent Neural Networks!

> 這邊算式有點多, 請看投影片

- RNN tie the weights at each time step.

#### Main idea : we use the same set of W weights at all time steps! 

- The class at each time step is just a word index of the next word.

- Same cross entropy loss function but predicting words instead of classes


---

# The vanishing gradient problem

- In the case of language modeling or question answering words from time steps far away are not taken in consideration when training to predict the next word

- Example 

- Jane walked into the room. John walked in too. It was late in the day. Jane said hi to __

---

# IPython Notebok with vanishing gradient example
# 

- Example of simple and clean NNet implementation

- Comparisino of sigmoid and Relu units

- A little bit of vanishing gradient

---


# Trick for exploding gradinet: clipping trick

- The solution first introduced by Mikolov is to clip gradients to a maximum value.

- Make a big difference in RNNs.

---

# Gradient clipping intuition

- Error surface of a single hiden unit RNN

- High curvature walls.

- Solid lines: standard gradient descent trajectories

- Dashed lines gradients rescaled to fixed size.


---

# For vanishing gradients: Initialization + ReLus!

- Initialize `W(*)'s` to identity martix i and f(z) = rect(z) = max(z, 0)

- -> Huge difference!

- Initializatino idea first introduced in Parsing with Compositional Vector Grammars, Socher et al. 2013

- New experiments with recurrent neural nets in A Simple Way to initialize Recurrent Networks of Rectified Linear Units, Le et al. 2015

---

# Problem: softmax is huge and slow

- Trick: Class-based word prediction

`p(Wt|history) = p(Ct|hostory)p(Wt|Ct) = p(Ct|ht)p(Wt|Ct)`


- The more classes, the better perplexity but also worse speed.

---


# One last implementation trick

- You only need to pass backwards through your sequence once and accumulate all the deltas from each Et.


---

# Sequence modeling for other tasks

- Classify each word into:
     - NER
     - Entity level sentiment in context
     - opinionated expressions

- Example application and slides from paper Opinion Mining with Deep Recurrent Nets by irsoy and Cardie, 2014

---

# Opinion Mining with Deep Recurrent Nets

- Goal: Classify eacj word as 

    - direct subjective expressions(DSEs) 
    - expressive subjective expressions(ESEs).

- DSE: Explicit mentions of private states or speech events expressing private states

- ESE: Expressions that indicate sentiment, emotion, etc.
without explicitly conveying them.


## Example 

- In BIO notation (tags either begin-of-entity(B_X) or continuation-of-entity(I_X)):

    - The committee, as usual, has refused to make any statements.

    - The committee, `as usual`(ESE), `has refused to make any statements`(DSE)



---

# Approach: Recurrent Neural Network

- Notation from paper(so you get used to different ones)


![](https://github.com/jneo8/CS224n/blob/master/images/l8_3.png?raw=true)

- x represents a token(word) as a vector.
- y represents the output label(B, I or O) - g = softmax !
- h is the memory, computed from the past memory and current word. It summarizes the sentence up to that time.


# Bidirectional RNNs

- problem: For classification you want to incorporate information from words both preciding and following


![](https://github.com/jneo8/CS224n/blob/master/images/l8_4.png?raw=true)

- h now represents(summarizes) the past adn future around a single token.

# Deep Bidrectional RNNs

![](https://github.com/jneo8/CS224n/blob/master/images/l8_5.png?raw=true)


- Each memory layer passes an intermediate sequential represention to the next.


---

# Recap 

- Recurrent Neural network is one of the best deepNLP model families

- Training them is hard because of vanishing and exploding gradient problems

- They can be extended in many ways and their training improved with many tricks(more to come)




































