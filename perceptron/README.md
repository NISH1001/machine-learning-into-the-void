# Perceptron

Perceptron is the building block for a larger network (which is called neural network to be taught later). 
It is a blackbox that accepts inputs, processes them and gives out outputs (actually tries to predict them).
A perceptron, in fact, is just a crude way to simulate a single biological neuron.
We know, a neuron fires (or does not fire) based on its input stimuli. So, a perceptron is like that:  
```bash
input ---> process ---> output (prediction)
```

# How does it work?
A perceptron is a decision maker. It outputs certain number which can be interpreted as **yes/no** or
**0/1** or similar range of decision.  

> For example, the prediction of a house cost depends on various input factors like area, resident type, 
popularity, number of rooms, etc. These inputs can be modeled as variables.

So,

```bash
|   x1  |
|   x2  |   ---> |process|  --->    |z|
|   x3  |
```
This is the overall blackbox model for a perceptron. 
It takes **x1, x2, x3** inputs go into the process. Does some *mathe-magic*. And, gives output.

## But wait a sec
Not all the inputs have influence over the final output/result. This is analogous to the biological neuron where 
some stimulus have greater influence on the resposne of the neuron.


> For example, to buy the house the weather doesn't really affect the pricing. 

So, the inputs are prioritized accordingly. These priorities are called **weights** in perceptron.
```bash
|   x1, w1  |
|   x2, w2  |   ---> |process|  --->    |z|
|   x3, w3  |
```

## Process
The process is nothing but weighted sum of inputs.

```bash
    x1*w1 + x2*w2 + x3*w3
```

In general sense,
```bash
    summation(xi, wi)
```

Vectoricallly,
```bash
    dot_product(X, W)
```

But, the summation doesn't tell us about decision because there has to be some conditions for decision making.  
Like: if **output/y** is more than some threshold, the perceptron fires. This very idea of threshold gives rise to 
some kind of *so-called* **activation** shit which we *machine learning freak* call **activation function**.

### So, what really happens is
```bash
    dot_product(X, W)   --->    y   ---> activation  --->    output(z)
```

The activation function accepts the weighted sum (which is a single value for given inputs) and performs some
conditions to fire up the neuron (or not to fire the neuron)

Normally, only simple conditions are used
```bash
    if y >= 0.6,    z = 1
    if y < 0.6,    z = 0
```

Or you can use already available activation function like [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)
This activation gives you the output in the range 0 to 1.
If this output is high around 1, it means perceptron has fired.

## Output
Strictly, perceptron outputs boolean value : **0/1** or **False/True**.
If you are using some kinda activation, the perceptron outputs some real numbers. 
But it isn't perceptron really. It is an artificial neural network.

> The difference between a neuron and perceptron is that a perceptron outputs boolean value while 
a neuron outputs a real number.

> For simplicity, I will consider both perceptron and a neuron as being same; used synonymously.

------

# Let's build a perceptron
We will be using `python3` and `numpy`. Other libraries/modules can be installed accordingly from `pip3`.

## Import numpy
Numpy is a python library for performing vector (matrix) operations efficiently.
```python
import numpy as np
```

## Training Inputs
The inputs are **numpy arrays**
```python
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
```

## Training Outputs
Let's try an **OR gate**. So, training output will be:

```python
>> Y_train = np.array([[0, 1, 1, 1]]).T
```

## Weights/Synapses
```python
>> synapses = np.array([ [ 2.45685895, 2.56862029] ]).T
```

This weight is actually obtained after training the perceptron. The above is what I have obtained after training for 
100 iterations.

## Sigmoid
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

## Prediction
```python
>> y = np.dot(X_train, synapses)
array([[ 0.        ],
       [ 2.56862029],
       [ 2.45685895],
       [ 5.02547924]])
>> z = sigmoid(y)
array([[ 0.5       ],
       [ 0.92881453],
       [ 0.92106159],
       [ 0.99347442]])
```

So, **OR Gate** is working nicely. The perceptron is giving us the output very high for any input that contains a **1**.

> Hence, perceptron is all about finding the real values of the weights.


## Training
In the above context, we have used already available weights from a trained system. 
But we don't get such *stable* weights always. And randomly guessing or doing brute-force solution is pretty much expensive (computationally).

Here, we just have **2 weights** for which brute force might work. But that's  just a childish imagination - having just few weights.

In real case scenario, we have **many** weights. I repeat **many**. 
So, what we do is start from random values of weights and use certain process to find our way to the weights that seem good enough for  the system.

### Random weights
```python
>> synapse = 2*np.random.random((2, 1)) - 1
array([[-0.67238843],
       [ 0.43981246]])
```

Using the random weights, we try the prediction.
```python
>> y = np.dot(X_train, synapses)
array([[ 0.        ],
       [ 0.43981246],
       [-0.67238843],
       [-0.23257597]])
>> z = sigmoid(y)
array([[ 0.5       ],
       [ 0.60821434],
       [ 0.33796224],
       [ 0.44211669]])
```

>> Oh shit! This is not good. 

Don't worry if the prediction is mayhem. We have a technique to learn the weights through error in the prediction.


### Errors
What we know is what the output should be for each training example from **Y_train**.
[Core gist of **supervised learning**]
So, we have a metric to how much off the predicted value is from what is expected (as from `Y_train`).

One of the ways for calculating the error is just using the difference:
```bash
error = target - prediction
```

In machine learning world, we call this a [cost function](https://stackoverflow.com/a/40445197/4287672)

```python
>> errors = Y_train - z
array([[ 0.5       ],
       [0.39178566],
       [0.66203776],
       [0.55788331]])
```


### Update
Using the error, we can know how much we should add/subtract to the corresponding weight in order to approach the target value.

1. If the error is positive, we have to add the error to the weight by that much amount.
2. If the error is negative, we have to subtract the error from the weight by that much amount.

```bash
new_weight -> old_weight + error
```

This is vaguely the rule for updating the weight

Generally,
```bash
wi = wi + error
```

However, the `error` factor isn't solely responsible for the weight. If error is alone responsible for the update, the learning is pretty much very slow.
Each weight is also contributed by the corresponding input it has connection to. So, in some ways inputs do influence the corresponding weights.

The intuition behind this **input** coming into the play is relatable.

> Say you have a metal rod. In normal condition, when you touch the rod it doesn't really have influence on your reaction.  
When you touch a **hot rod**, you immediately withdraw you hand. So, in some ways the synapse is wired to respond to the
*hotness* or *coldness* of the rod itself.  The material from which rod is made can be considered as weight here.

So,
```bash
your_reaction -> (coldness/hotness of rod) + (rod's material)
```

This intuition is just a naive one which I have thought of.

Remember this the whole time or on every machine learing processes:
> Inputs affect the outputs.

#### Back to square one - Let's update the weight again.

```bash
wi = wi + error * input
```

So,
```bash
w1  ->  w1 + error * x1
w2  ->  w2 + error * x2
```

For each training set (X_train, Y_train):

```bash
w1  -> w1 + error1 * x1 + error2 * x1 + ...     ==> w1 + (error1 + error2 +...) * x1
w2  -> w2 + error1 * x2 + error2 * x2 + ...     ==> w2 + (error1 + error2 +...) * x2
```

Generally,
```bash
wi  -> wi + average_error * xi
```

### Learning rate
*So far so good, heh?*  

Yes. But no. Here's the problem. If we did follow the above rule, then we run into one specific problem.

We could just keep on oscillating here and there. At one time the error might be positive, at another time it might be negative.
The weights might just be like a pendulum and their values oscillating here and there.

So, we introduce a parameter ( [Hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter) that directly affects this behaviour.
One of such parameters is learning rate that tells by how much factor the  weights have to be updated to get to the **optimal** configuration.

There are two things that matter:

1. If learning rate is too big, we might miss the optimal configuration. We might be oscillating here and there.
2. If learning rate is too small, it might take eternity to reach the desired weights.

> Learning rate is generally represented by Greek letter called *eta*.

```python
>> eta = 0.01
```

### Perceptron Learning Rule (Finally)

```bash
wi  ->  wi + eta * average_error * xi
```

```python
>> delta = np.dot( X_train.T, errors)
array([[ 1.21992107],
       [ 0.94966897]])
```

```python
>> synapses +=  eta * delta
array([[-0.55039632],
       [ 0.53477936]])
```

### Final Note
Run this update rule for many iterations and you'll get the optimial weights.

>> The *optimial weight* I have been mentioning is just one of the configurations of weights for which the model accurately predicts the output.
However it is not guaranteed that the system/model converges to a global optmial configuration. This is the intuition behind [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
