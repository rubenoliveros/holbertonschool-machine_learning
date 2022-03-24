#!/usr/bin/env python3
"""4. Continuous Posterior"""


from scipy import special


def posterior(x, n, p1, p2):
    """
    A function that calculates the posterior probability that the
    probability of developing severe side effects falls within a
    specific range given the data
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(p1) is not float or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if type(p2) is not float or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    # use p and (1 - p) or in our case x and (n - x) for the data
    # add uniformly distributed priors with +1 in parameters
    # cumulative distribution function for beta distribution for 0 to p1
    beta_dist1 = special.btdtr(x + 1, n - x + 1, p1)
    # cumulative distribution function for beta distribution for 0 to p2
    beta_dist2 = special.btdtr(x + 1, n - x + 1, p2)
    # subtract to get the difference between p2 and p1
    posterior = beta_dist2 - beta_dist1
    return posterior
