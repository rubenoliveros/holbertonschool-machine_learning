#!/usr/bin/env python3
"""4. The Viretbi Algorithm"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden
       markov model
    Args:
        Observation is a numpy.ndarray of shape (T,) that contains the index of
                    the observation
                    - T is the number of observations
        Emission is a numpy.ndarray of shape (N, M) containing the emission
                 probability of a specific observation given a hidden state
                 - Emission[i, j] is the probability of observing j given the
                                  hidden state i
                 - N is the number of hidden states
                 - M is the number of all possible observations
        Transition is a 2D numpy.ndarray of shape (N, N) containing the
                   transition probabilities
                   - Transition[i, j] is the probability of transitioning from
                                      the hidden state i to j
        Initial is a numpy.ndarray of shape (N, 1) containing the probability
                of starting in a particular hidden state
    Returns: path, P, or None, None on failure
        - path is the a list of length T containing the most likely sequence of
               hidden states
        - P is the probability of obtaining the path sequence
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

    V = np.zeros((n, t))
    BP = np.zeros((n, t))

    V[:, 0] = Initial.transpose() * Emission[:, Observation[0]]

    for ti in range(1, t):
        ab = V[:, ti - 1] * Transition.T
        ab_m = np.amax(ab, axis=1)
        prob = ab_m * Emission[:, Observation[ti]]
        V[:, ti] = prob
        BP[:, ti - 1] = np.argmax(ab, axis=1)

    path = []
    current = np.argmax(V[:, t - 1])
    path = [current] + path

    for ti in range(t - 2, -1, -1):
        current = int(BP[current, ti])
        path = [current] + path

    P = np.amax(V[:, t - 1], axis=0)

    return path, P
