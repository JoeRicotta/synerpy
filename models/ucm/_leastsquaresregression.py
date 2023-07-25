import numpy as np

from sklearn.linear_model import LinearRegression

from ._base import _BaseUCM

class _LeastSquaresRegression(LinearRegression, _BaseUCM):

    def fit(self, X, y):

        #fitting regression
        super().fit(X, y)
        if len(self.coef_.shape) == 2:
            rows, cols = self.coef_.shape
            if rows > cols:
                self.jacobian_ = self.coef_.reshape(cols, rows)
            else:
                self.jacobian_ = self.coef_
            
        self.scores_ = X

        # fitting ucm
        self._ucm_fit()

        return self
