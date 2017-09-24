# Structured Trainging for Neural Network Transitino-Based Parsing


## What is SyntaxNet?


- 2016/5:
    - Google announces the "World's Most Accurate Parser Goes Open Source"
- SyntaxNet(2016): New, fast, performant Tensorflow framework for syntactic parsing.

- Now supports 40 languages -- Parse McParseface's 40 'cousins'


## 3 New contributions


- Leverage Unlabelled Data -- "Tri-training"


- Turnd Neural Network Model.

- Final Layer: Structured Perceptron w / Beam Search
    - The most important


## Tri-Training 
   
- Agree on dependency parse or Disagree on dependency parse.


## Model change

- Wess et al(2015)
    - input data ->  2X (hidden layer, RELU) -> Softmax Layer -> Perceptron Layer


## Structured Perceptron Training + Beam Search

- Problem: Gready algorithm are unable to look beyond one step ahead, or recover from incorrect decision.

- Solution: Look forward -- search the tree of possible transition sequences.
    - Keep track of `K` top partial transition sequences up to depth `m`.





    















