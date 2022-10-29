"""
a few helpful features and functions used in the regression case.
"""
from numpy import mean

from ..utils.validation import _check_y


def r_squared(y, y_pred):
    """
    takes y, and predictor, and returns the r-squared value.
    """
    y, y_pred = _check_y(y), _check_y(y_pred)
    
    sse = sum((y-y_pred)**2)
    sst = sum((y - mean(y))**2)

    return 1 - (sse/sst)
