from ._base import BaseModel
from ..base import TransformerMixin

## TODO:
## PUT FACTOR ANALYSIS AND PCA IN SEPARATE FILES


class Modes(BaseModel, TransformerMixin):
    """
    Typically defined for MU data
    Madarshahian et al 2021

    Please consider that the results of interest are likely sensitive to whether the data used for the results
    is scaled or not scaled. Condsider Modes()._use_scaled_ = True for the analyses.
    """

    def __init__(self, n_factors = None, rotation = None, eval_crit = lambda x: True, loading_crit = lambda x: True):
        super().__init__()
        self.eigenvalues = None
        self.loadings = None
        self.score = None
        self.rotation_mtx = None
        self.n_factors = n_factors
        self.rotation = rotation
        self.max_iter = 500
        self.tol = 1e-6
        self.var_exp = None
        self.eval_crit = eval_crit
        self.loading_crit = loading_crit
        self.magnitudes = None

    
    def _fit(self, X, y = None):
        """
        y ignored.
        Will perform factor analysis using principal component method,
        with or without varimax rotation.

        The code for this function was adapted directly from factor_analyzer package
        """
        # perform principal component analysis on the data
        # simplified version of factor_analysis from factor_analyzer package
        # https://github.com/EducationalTestingService/factor_analyzer

        # and to get them after rotation, use commonalities (https://online.stat.psu.edu/stat505/lesson/12/12.11o

        # getting n_factors
        if self.n_factors is None:
            # make all principal components
            n_factors = X.shape[1]
        else:
            n_factors = self.n_factors
        
        # normalizing data cov matrix
        if self._use_scaled_:
            Cx = np.corrcoef(X.T)
        else:
            Cx = np.cov(X.T)

        # total variance
        tot_var = sum(np.diag(Cx))
        
        # getting eigenvalues and eigenvectors using hermitian eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(Cx)
        eigenvalues, eigenvectors = eigenvalues[::-1][0:n_factors], eigenvectors[:,::-1][:,0:n_factors]

        # getting variance explained by each factor
        var_exp = eigenvalues / tot_var
        loadings = eigenvectors * np.sqrt(eigenvalues)
        n_rows, n_cols = loadings.shape
        
        # initialize the rotation matrix. Will remain N x N identity matrix if
        # varimax rotation not called-- otherwise, will be iteratively
        # updated in the below code.
        rotation_mtx = np.eye(n_cols)

        # if varimax rotation is requested, create rotation matrix.
        if self.rotation == "varimax" and n_cols >= 2:
            
            L = loadings.copy()

            # normalize the loadings matrix
            # using sqrt of the sum of squares (Kaiser)
            if self._use_scaled_:
                normalized_mtx = np.apply_along_axis(
                    lambda x: np.sqrt(np.sum(x**2)), 1, L.copy()
                )
                L = (L.T / normalized_mtx).T

            # index and rotation matrix
            d = 0
            for _ in range(self.max_iter):

                old_d = d

                # take inner product of loading matrix
                # and rotation matrix
                basis = np.dot(L, rotation_mtx)

                # transform data for singular value decomposition using updated formula :
                # B <- t(x) %*% (z^3 - z %*% diag(drop(rep(1, p) %*% z^2))/p)
                diagonal = np.diag(np.squeeze(np.repeat(1, n_rows).dot(basis**2)))
                transformed = L.T.dot(basis**3 - basis.dot(diagonal) / n_rows)

                # perform SVD on
                # the transformed matrix
                U, S, V = np.linalg.svd(transformed)

                # take inner product of U and V, and sum of S
                rotation_mtx = np.dot(U, V)
                d = np.sum(S)

                # check convergence
                if d < old_d * (1 + self.tol):
                    break

            # take inner product of loading matrix
            # and rotation matrix
            L = np.dot(L, rotation_mtx)

            # de-normalize the data
            if self._use_scaled_:
                L = L.T * normalized_mtx
            else:
                L = L.T

            # convert loadings matrix to data frame
            loadings = L.T.copy()

            # resetting eigenvalues after rotation
            eigenvalues = np.apply_along_axis(lambda x: np.sum(x**2), 0, loadings)
            
        # filter
        inds = self.eval_crit(eigenvalues) * np.apply_along_axis(self.loading_crit, 0, loadings)

        # assigning attribute values
        self.loadings = loadings.T[inds].T
        self.rotation_mtx = rotation_mtx
        self.eigenvalues = eigenvalues[inds]      
        self._is_fitted_ = True
        self.var_exp = eigenvalues / tot_var
        self.tot_var = tot_var

        return self


    def _transform(self, X, y = None):
        """
        Ignores y input.
        """

        # check to see if model has been fit
        self._check_is_fitted()

        # check dimension match between jacobian and X
        if self.loadings.shape[0] != X.shape[1]:
            raise(ValueError(f"Data matrix of different dimension ({X.shape[1]}) than loadings ({self.loadings.shape[0]}). Use a different data matrix."))

        # using demeaned X
        X_dem = self._rm_mean(X)
        
        # get projections
        self.magnitudes = X_dem @ self.loadings

        return self

    
    def fit_transform(self, X, y = None):
        """
        Ignores y input.
        """
        self._fit(X)._transform(X)

        return self
