#!/usr/bin/env python3
"""1. Privatize Neuron"""
import numpy as np


class Neuron:
    """a class that defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """Initializes the class"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Function for the weights vector for the neuron"""
        return self.__W

    @property
    def b(self):
        """Function for the bias for the neuron"""
        return self.__b

    @property
    def A(self):
        """Function for the activated output of the neuron (prediction)"""
        return self.__A
