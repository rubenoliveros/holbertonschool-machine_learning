#!/usr/bin/env python3
"""18. DeepNeuralNetwork Forward Propagation"""
import numpy as np


class DeepNeuralNetwork:
    """a class that defines a deep neural network"""
    def __init__(self, nx, layers):
        """Initialize the class"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        layer = 1
        layer_size = nx
        for i in layers:
            if type(i) != int or i <= 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W" + str(layer)
            b = "b" + str(layer)
            self.weights[w] = np.random.randn(
                i, layer_size) * np.sqrt(2/layer_size)
            self.weights[b] = np.zeros((i, 1))
            layer += 1
            layer_size = i

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            w = "W" + str(i)
            b = "b" + str(i)
            a = "A" + str(i - 1)
            Z = np.matmul(self.__weights[w],
                          self.__cache[a]) + self.__weights[b]
            a_new = "A" + str(i)
            self.__cache[a_new] = 1 / (1 + np.exp(-Z))
        Act = "A" + str(self.__L)
        return (self.__cache[Act], self.__cache)

    @property
    def L(self):
        """The number of layers in the neural network"""
        return self.__L

    @property
    def cache(self):
        """A dictionary to hold all intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """A dictionary to hold all weights and biased of the network"""
        return self.__weights
