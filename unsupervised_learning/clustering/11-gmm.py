#!/usr/bin/env python3
"""module GMM"""
import sklearn.mixture


def gmm(X, k):
    """calculates a gaussian misture Model from a dataset"""
    gmm = sklearn.mixture.GaussianMixture(n_components=k)
    gmm.fit(X)

    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
