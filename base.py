"""
Base class for all estimators.
Synerpy has borrowed the code architecture and philosophy from scikit-learn:
(https://github.com/scikit-learn/scikit-learn/)
If you haven't heard of sklearn, now is the best time to.
"""
from .utils.validation import (
    _check_is_fitted,
    _check_X,
    _check_X_y,
    _check_y)

class BaseEstimator:

    def check_is_fitted(self):
        return _check_is_fitted(self)

    @staticmethod
    def check_X(X):
        return check_X(X)

    @staticmethod
    def check_y(y):
        return check_y(y)

    @staticmethod
    def check_X_y(X, y):
        return _check_X_y(X, y)

    
class TransformerMixin:

    def _transform(self, X, y = None):
        return self.transform(X,y)

