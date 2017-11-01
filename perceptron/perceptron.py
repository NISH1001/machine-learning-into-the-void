#!/usr/bin/env python3

import numpy as np
#np.random.seed(1)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class HyperParameters:
    """
        Parameters that affect overall behaviour
    """
    def __init__(self, alpha):
        self.alpha = alpha

class Perceptron:
    """
        The basic unit that simulates a single neuron.
            input --> neuron --> output

        The Output is actually the weighted sum of input vector

        A perceptron is basically a neuron that fires or doesn't fire.
    """

    def __init__(self, input_vector_size, hyperparams, activation, epoch=200):
        self.input_vector_size = input_vector_size
        self.hyperparams = hyperparams
        self.activation = activation
        self._epoch = epoch
        self.synapse = 2*np.random.random((input_vector_size, 1)) - 1
        self.synapse_with_bias = 2*np.random.random((input_vector_size + 1, 1)) - 1

    def train(self, X_train, Y_train):
        """
            Feed-forward training
        """
        bias = np.ones( Y_train.shape )
        X_train_with_bias = np.append(X_train, bias, axis=-1)
        for i in range(self._epoch):
            Y = np.dot(X_train_with_bias, self.synapse_with_bias)
            Z = self.activation(Y)
            delta = np.dot(X_train_with_bias.T, (Y_train - Z ))
            self.synapse_with_bias += self.hyperparams.alpha * delta

    def predict(self, x):
        x_with_bias = np.append(x, np.ones((1)))
        return self.activation(np.dot(x_with_bias, self.synapse_with_bias))

def test_perceptron():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_train = np.array([[0, 1, 1, 1]]).T

    hyperparams = HyperParameters(0.1)
    perceptron = Perceptron(2, hyperparams, sigmoid, 3000)
    perceptron.train(X_train, Y_train)
    print(perceptron.predict(np.array([0, 0])))

def main():
    test_perceptron()

if __name__ == "__main__":
    main()
