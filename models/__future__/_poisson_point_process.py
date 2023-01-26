"""
A model for handling poisson point process estimation of spike data.
"""
import numpy as np

from sklearn.base import BaseEstimator

from ...datasets import load_firings

firings = load_firings()
data = firings.data

class _Poisson_Point_Process(BaseEstimator):
    """
    Modeling the intensity function of the poisson point process
    as the expected value of a gamma-distributed random variable
    parameter of a continuous poisson process.
    """

    def __init__(self, time_window=None):
        pass
    # if time_window is None, auto-estimate time window.

    def fit(self, X, y = None):
        pass

    
