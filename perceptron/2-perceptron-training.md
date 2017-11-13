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

