# Introduction to tensorflow

---


## Programming model

Big idea: express a numeric computation as a `graph`.

- Graph nodes are operations which have any number of inputs and outputs.
- Graph edges are tensors which flow between nodes. 

---

![](https://github.com/jneo8/CS224n/blob/master/images/l7_1.png?raw=true)

- `Variables` are stateful nodes which output their current value

	- In this case, it's just b and W what we mean by saying that variables.

	- State is retained across multiple executions of a graph. (mostly parameters)

- `placeholder` are nodes whose value is fed in at execution time 
	- inputs, labels
	- In this case is X.

- `Mathematical operations`

	- MatMul: multiply two matrix values.
	- Add: Add elementwise(with broadcasting)
	- ReLu: Activate with elementwise rectified linear function.

	
### In code

1. Create weights, including initialization
 
 	- W ~ Uniform(-1, 1); b=0
 
2. Create input placeholder x, m * 784 input matrix
3. Build flow graph

```python
"""Relu example."""
import tenforflow as tf
import numpy as np

b = tf.Variable(tf.zeros((100,)))
w = tf.Variable(tf.random_uniform((784, 100), -1, 1))
x = tf.placeholder(tf.float32, (100, 784))
h = tf.nn.relu(tf.matmul(x, w) + b)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(h, {x: np.random.random((100, 784))})
```

### How do we run it?
So far we have defined a graph.

We can deploy this graph with `session`:
a binding to a particular execution context.(e.g. CPU, GPU)


---

## How do we define the loss?
Use placeholder for labels
Build loss node using labels and prediction

```python
prediction = tf.nn.softmax(...)  # Output of neural network
label = tf.placeholder(tf.float32, [100, 10])

cross_entropy = -tf.reduce_sum(label * tf.log(prediction), axis=1)
```

---

## How do we compute Gradients

```
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

- tf.train.GradientDescentOptimizer is an `Optimizer` object.
- tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy) adds optimization `operation` to computation graph.


**TensorFlow graph nodes have attached gradient operations Gradient with respect to parameters computed with backpropagation**


## training the model

```python

prediction = tf.nn.softmax(...)label = tf.placeholder(tf.float32, [None, 10])cross_entropy = tf.reduce_mean(
	-tf.reduce_sum(label * tf.log(prediction),	reduction_indices=[1])
)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# training the model
sess = tf.Session()sess.run(tf.initialize_all_variables())for i in range(1000):     batch_x, batch_label = data.next_batch()     sess.run(
     	train_step,
     	feed_dict={
     		x: batch_x,         	label: batch_label
       },
    )
```


---

## Value scope

```python
tf.variable_scope()  # provides simple name-spacing to avoid clashes.
tf.get_variable()  # creates/accesses variables from within a variable scope.


with tf.variable_scope('foo'):
	v  = tf.get_variable('v', shape=[1])  # v.name == "foo/v:0"
	
with tf.variable_scope('foo', reuse=True):
	v1 = tf.get_variable('v')  # shared variable found!
	
with tf.variable_scope('foo', reuse=False):
	v1 = tf.get_variable('v')  # CRASH foo/v:0 already exists!
```

---

## In summary

- Build a graph
	- Feedforward / Prediction
	- Optimization

- Initialize a session

- Train with session.run(train_step, feed_dict)







