#!/usr/bin/env python3
"""10. NeuralNetwork Forward Propagation"""
import numpy as np


class NeuralNetwork:
    """a class that defines a neural network with one hidden layer"""
    def __init__(self, nx, nodes):
        """Initializes the class"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    def forward_prop(self, X):
        """iCalculates the forward propagation of the neural network"""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return (self.__A1, self.__A2)

    @property
    def W1(self):
        """Function for the weights vector for the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Function for the bias for the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Function for the activated output for the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Function for the weights vector for the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Function for the bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Function for the activated output for the output neuron"""
        return self.__A2
