#!/usr/bin/env python3
"""module hello sklearn"""
import numpy as np
import sklearn.cluster


def kmeans(X, k):
    """Performs k-means clustering on a dataset."""
    kmeans_ = sklearn.cluster.KMeans(n_clusters=k)
    kmeans_.fit(X)
    C = kmeans_.cluster_centers_
    clss = kmeans_.labels_

    return C, clss
