#!/usr/bin/env python3
"""11. GMM"""

import sklearn.mixture


def gmm(X, k):
    """
    this calculates the gmm of a dataset
    returns:
    a list containing the cluster centroids, cluster assignments,
    pi, and the gaussian distribution parameters
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=k)

    gmm_resp = gmm.fit(X)

    clss = gmm.predict(X)

    pi = gmm_resp.weights_
    m = gmm_resp.means_
    S = gmm_resp.covariances_

    return pi, m, S, clss, gmm.bic(X)
