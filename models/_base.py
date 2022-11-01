"""
base functionality for models.
"""

import numpy as np

from sklearn.utils.validation import check_array


def varimax(X, _use_scaled_ = False, max_iter = 500, tol = 1e-6):

    X = check_array(X)

    n_rows, n_cols = X.shape
    if n_cols < 2:
        return X

    X = X.copy()

    # normalize the loadings matrix
    # using sqrt of the sum of squares (Kaiser)
    if _use_scaled_:
        normalized_mtx = np.apply_along_axis(
            lambda x: np.sqrt(np.sum(x**2)), 1, X.copy()
        )
        X = (X.T / normalized_mtx).T


    rotation_mtx = np.eye(n_cols)

    # index and rotation matrix
    d = 0
    for _ in range(max_iter):

        old_d = d

        # take inner product of loading matrix
        # and rotation matrix
        basis = np.dot(X, rotation_mtx)

        # transform data for singular value decomposition using updated formula :
        # B <- t(x) %*% (z^3 - z %*% diag(drop(rep(1, p) %*% z^2))/p)
        diagonal = np.diag(np.squeeze(np.repeat(1, n_rows).dot(basis**2)))
        transformed = X.T.dot(basis**3 - basis.dot(diagonal) / n_rows)

        # perform SVD on
        # the transformed matrix
        U, S, V = np.linalg.svd(transformed)

        # take inner product of U and V, and sum of S
        rotation_mtx = np.dot(U, V)
        d = np.sum(S)

        # check convergence
        if d < old_d * (1 + tol):
            break

    # take inner product of loading matrix
    # and rotation matrix
    X = np.dot(X, rotation_mtx)

    # de-normalize the data
    if _use_scaled_:
        X = X.T * normalized_mtx
    else:
        X = X.T

    # convert loadings matrix to data frame
    loadings = X.T.copy()

    return loadings, rotation_mtx
