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
































