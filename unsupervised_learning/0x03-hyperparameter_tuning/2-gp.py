#!/usr/bin/env python3
"""2. Update Gaussian Process"""
import numpy as np


class GaussianProcess():
    """A class that represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Represents a noiseless 1D Gaussian process"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix"""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1)
        sqdist2 = sqdist - 2 * np.dot(X1, X2.T)
        result = self.sigma_f ** 2 * np.exp(-0.5 / self.l**2 * sqdist2)
        return result

    def predict(self, X_s):
        """Predicts the mean and standard deviation"""
        K1 = self.kernel(self.X, X_s)
        K2 = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K1.T.dot(K_inv).dot(self.Y)
        cov = K2 - K1.T.dot(K_inv).dot(K1)
        return mu.reshape(-1), cov.diagonal()

    def update(self, X_new, Y_new):
        """Updates a Gaussian Process"""
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
