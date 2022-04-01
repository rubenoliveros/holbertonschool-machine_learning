#!/usr/bin/env python3
"""12. Agglomerative"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    this function performs agglomerative clustering
    Args:
        X ([type]): [description]
        dist ([type]): [description]
    Returns:
        [type]: [description]
    """

    Z = scipy.cluster.hierarchy.linkage(
        X,

        method='ward'
    )

    _ = plt.figure(figsize=(25, 10))
    _ = scipy.cluster.hierarchy.dendrogram(
        Z,
        color_threshold=dist
    )
    plt.show()

    return scipy.cluster.hierarchy.fcluster(
        Z,
        t=dist,
        criterion='distance'
    )
