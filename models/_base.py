"""
Base class for all models.
In the framework for synerpy, all models are objects with fit methods.
THis means they take in data and produce some model of that data
given some assumptions about the connections of that data.
"""

from ..base import BaseEstimator, TransformerMixin
from numpy import mean, std


class BaseModel(BaseEstimator):

    def __init__(self, _is_fitted_ = False, _use_scaled_ = False):
        self._is_fitted_ = _is_fitted_
        self._use_scaled_ = _use_scaled_

    # all models must, by definition, have a fit function, although they need not
    # have a transform function as well.
    def fit(self, X, y = None):
        """
        Checks data input and returns whether or not structure is fit.
        """
        if y is not None:
            X, y = self.check_X_y(X,y)
        else:
            X = self.check_X(X)

        # pass self fit function
        return self._fit(X,y)
        
    @staticmethod
    def _rm_mean(X):
    """
    returns dataframe centered at 0 mean.
    """
        mean_ = mean(X, axis = 0)
        
        return X - mean_

    
    def _standardize_data(self, X):
        """
        Transforms data to scale to 0 mean and unit variance.
        """
        return self._rm_mean(X) / np.std(X, axis = 0)
