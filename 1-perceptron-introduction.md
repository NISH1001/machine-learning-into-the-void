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
