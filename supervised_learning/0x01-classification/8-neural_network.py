#!/usr/bin/env python3
"""8. NeuralNetwork"""
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
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
