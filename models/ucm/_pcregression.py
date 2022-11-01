import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from ._base import _BaseUCM

# API to sklearn regression and sklearn PCA.
# Should be safe wrt version updates.

class _PCAMixin(PCA):

    def _fit_transform_pca(self, X, y = None):
        """
        renaming pca method to avoid aliasing.
        """
        out = super().fit_transform(X,y)
        return out

class _PCRegression(LinearRegression, _PCAMixin, _BaseUCM):

    def __init__(self, n_components = None):
        # adding PC parameters manually
        self.__dict__.update(
            _PCAMixin().get_params()
        )
        self.n_components = n_components                
        super().__init__()
        
    def fit(self, X, y):
        
        # fitting PCs
        U = self._fit_transform_pca(X)
        self.scores_ = U

        # fitting regression
        super().fit(U, y)
        # storing jacobian as 2d
        self.jacobian_ = np.array([self.coef_])

        # fiting ucm
        self._ucm_fit()

        return self


