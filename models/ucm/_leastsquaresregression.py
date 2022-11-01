# LSRegMixin
from sklearn.linear_model import LinearRegression

from ._base import _BaseUCM

class _LeastSquaresRegression(LinearRegression, _BaseUCM):

    def fit(self, X, y):

        #fitting regression
        super().fit(X, y)
        self.jacobian_ = np.array([self.coef_])
        self.scores_ = X

        # fitting ucm
        self._ucm_fit()

        return self
