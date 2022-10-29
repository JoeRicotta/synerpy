# methods to check data
def _check_X(X):
    """
    Performs a check on X to make sure the covariance matrix is not singular.
    Will demean the data and scale to unit variance if _use_scaled_ is true.
    """
    # asserting shape and size
    assert isinstance(X, np.ndarray) and len(X.shape) == 2, "Data matrix X must be np.array() with 2 dimensions."

    # checking size of matrix
    if X.shape[1] > X.shape[0]:
        warnings.warn(f"\nData matrix X is long ({X.shape[0]} rows < {X.shape[1]} cols), and the covariance matrix is singular. This may lead to weird behavior.\n")

    return X

def _check_y(y):
    """
    Performs a check on the response variable formatting.
    """
    # checking dimensionality is appropriate
    assert isinstance(y, np.ndarray) and len(y.shape) == 1, "Response vector y must be np.array() with 1 dimension"

    return y

def _check_X_y(X,y):
    """
    performs checks on both X and y and ensures both are the same size in at least one dimension,
    """
    X = _check_X(X)
    y = _check_y(y)
    assert y.shape[0] in X.shape, f"X and y arrays are of non-matching size (X: {X.shape}, y: {y.shape})"

    return X, y

def _check_is_fitted(estimator):

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not estimator._is_fitted_:
        raise ValueError("estimator has not been fitted yet.")
    
