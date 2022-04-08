#!/usr/bin/env python3
"""3. The Forward Algorithm"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model
    Args:
        Observation is a numpy.ndarray of shape (T,) that contains the index of
                    the observation
            - T is the number of observations
        Emission is a numpy.ndarray of shape (N, M) containing the emission
                 probability of a specific observation given a hidden state
            - Emission[i, j] is the probability of observing j given the hidden
                             state i
            - N is the number of hidden states
            - M is the number of all possible observations
        Transition is a 2D numpy.ndarray of shape (N, N) containing the
                   transition probabilities
        Transition[i, j] is the probability of transitioning from the hidden
                         state i to j
        Initial a numpy.ndarray of shape (N, 1) containing the probability of
                starting in a particular hidden state
    Returns: P, F, or None, None on failure
        - P is the likelihood of the observations given the model
        - F is a numpy.ndarray of shape (N, T) containing the forward path
            probabilities
        - F[i, j] is the probability of being in hidden state i at time j
                  given the previous observation
    """
    if (type(Observation) is not np.ndarray or len(Observation.shape) != 1):
        return None, None

    t = Observation.shape[0]

    if (type(Emission) is not np.ndarray or len(Emission.shape) != 2):
        return None, None

    n, m = Emission.shape

    if (type(Transition) is not np.ndarray or Transition.shape != (n, n)):
        return None, None

    if (type(Initial) is not np.ndarray or Initial.shape != (n, 1)):
        return None, None

    F = np.empty((n, t))

    for i in range(t):
        if (i == 0):
            F[:, i] = Emission[:, Observation[i]] * Initial.T
        else:
            F[:, i] = (Emission[:, Observation[i]] *
                       np.dot(F[:, i - 1], Transition))

    likelihood = np.sum(F[:, t - 1])

    return likelihood, F
