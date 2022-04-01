#!/usr/bin/env python3
"""10. Hello, sklearn!"""

import sklearn.cluster


def kmeans(X, k):
    """
    This use the library scikit-learn to do K-means clustering
    return:
        clf: the trained model
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)

    return kmeans.cluster_centers_, kmeans.labels_
