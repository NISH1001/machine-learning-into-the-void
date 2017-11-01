# Perceptron

Perceptron is the building block for a larger network (which is called neural network to be taught later). 
It is a blackbox that accepts inputs, processes them and gives out outputs (actually tries to predict them).
A perceptron, in fact, is just a crude way to simulate a single biological neuron.
We know, a neuron fires (or does not fire) based on its input stimuli. So, a perceptron is like that:  
```bash
input ---> process ---> output (prediction)
```

## How does it work?
A perceptron is a decision maker. It outputs certain number which can be interpreted as **yes/no** or
**0/1** or similar range of decision.  

> For example, the prediction of a house cost depends on various input factors like area, resident type, 
popularity, number of rooms, etc. These inputs can be modeled as variables.

So,

```bash
|   x1  |
|   x2  |   ---> |process|  --->    |y|
|   x3  |
```
This is the overall blackbox model for a perceptron. 
It takes **x1, x2, x3** inputs go into the process. Does some *mathe-magic*. And, gives output.

### But wait a sec
Not all the inputs have influence over the final output/result. This is analogous to the biological neuron where 
some stimulus have greater influence on the resposne of the neuron.


> For example, to buy the house the weather doesn't really affect the pricing. 

So, the inputs are prioritized accordingly. These priorities are called **weights** in perceptron.
```bash
|   x1  x2  x3  | are weighted
```
