#!/usr/bin/env python3
"""7. Upgrade Train Neuron"""
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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """iCalculates the cost of the model using logistic regression"""
        cost_array = np.multiply(np.log(A), Y) + np.multiply((
            1 - Y), np.log(1.0000001 - A))
        cost = -np.sum(cost_array) / len(A[0])
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        return (np.where(self.__A > 0.5, 1, 0), cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        cost = self.cost(Y, A)
        dz = A - Y
        dw = (1 / len(Y[0])) * np.matmul(dz, X.T)
        db = (1 / len(Y[0])) * np.sum(dz)
        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the neuron by updating the attributes __W, __b, and __A"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        iters = []
        for i in range(iterations + 1):
            A, cost = self.evaluate(X, Y)
            if i != iterations:
                self.forward_prop(X)
                self.gradient_descent(X, Y, self.__A, alpha)
            if ((i % step == 0 or i == 0 or i == iterations) and
                verbose is True):
                print("Cost after {} iterations: {}".format(i, cost))
                costs.append(cost)
                iters.append(i)
        if graph is True:
            plt.plot(iters, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.xlim(0, iterations)
            plt.show()
        return (A, cost)

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
