# core package for all models.
# all models take in data and either
# 1. fit it,
# 2. transform it, or
# 3. fit and/or transform

import numpy as np
from scipy import linalg
import warnings

X = np.array([[1,2,3,4],
              [4,8,9,6],
              [8,0,7,5],
              [8,4,1,0],
              [2,2,3,2]])

y = np.array([4,8,5,6,0])

class _Model(object):

    def __init__(self, score = None, _is_fitted_ = False, _use_scaled_ = False):
        self.score = score
        self._is_fitted_ = _is_fitted_
        self._use_scaled_ = _use_scaled_

    @staticmethod
    def _scale_data(X):
        """
        Transforms data to scale to 0 mean and unit variance.
        """
        return (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)

    # methods to check data
    def _check_X(self, X):
        """
        Performs a check on X to make sure the covariance matrix is not singular.
        Will demean the data and scale to unit variance if _use_scaled_ is true.
        """
        # asserting shape and size
        assert isinstance(X, np.ndarray) and len(X.shape) == 2, "Data matrix X must be np.array() with 2 dimensions."
   
        # checking size of matrix
        if X.shape[1] > X.shape[0]:
            warnings.warn(f"\nData matrix X is long ({X.shape[0]} rows < {X.shape[1]} cols), and the covariance matrix is singular. This may lead to weird behavior.\n")

        # if scaling is desired, make mean 0 with unit variance
        if self._use_scaled_:
            X = self._scale_data(X)
           
        return X

    def _check_y(self, y):
        """
        Performs a check on the response variable formatting.
        """
        # checking dimensionality is appropriate
        assert isinstance(y, np.ndarray) and len(y.shape) == 1, "Response vector y must be np.array() with 1 dimension."

        # checking scaling
        if self._use_scaled_:
            y = self._scale_data(y)
           
        return y

    def _check_is_fitted(self):
        """
        Test to see if model has been fit.
        """
        if not self._is_fitted_:
            raise(ValueError("transform() called before fit()"))



#################################################
############## UCM ##############################
#################################################

class _UCMmodel(_Model):
    """
    class with common elements of all UCM analyses.
    """

    def __init__(self):
        super().__init__(None, False, False)
        self.j = None
        self.dv = None
        self.onb = None
        self.dim_ort = None
        self.dim_ucm = None
        self.projections = None
        self.vucm = None
        self.vort = None
    
    @staticmethod
    def _check_j(j):
        """
        checks features of the jaconbian and fixes them when possible.
        The jacobian matrix should be a wide matrix, or a square matrix.
        A long jacobian matrix means that the system is probably overdetermined, and the solution
        (if it exists) is singular.
        """
        # make sure j was passed to ucm
        if j is None:
            raise(ValueError("No value of the jacobian passed to UCM()."))
        
        # checking for proper dimension of jacobian
        assert isinstance(j, np.ndarray), f"Jacobian j={j} must be a 2d numpy ndarray."

        # making sure jacoobian is a 2-d matrix (not just vector):
        if len(j.shape) == 1:
            j = np.array([j])

        # making sure jacobian is properly shaped
        # as a wide or square matrix
        if j.shape[0] > j.shape[1]:
            j = j.T

        # normalizing each row vector
        norm = np.diag((1 / linalg.norm(j, axis = 1)))
        j = norm @ j

        return j

    @staticmethod
    def _def_onb(j):
        """
        Takes a jacobian and defines an orthonormal basis of ORT and UCM.
        Returns the orthonormal basis as well as the dimensions of ORT and UCM.
        """
        # getting basis of nullspace of the jacobian matrix (ucm) and dimension
        ucm = linalg.null_space(j)
        dim_ucm = ucm.shape[1]

        # getting ort from ucm and dimension
        ort = linalg.null_space(ucm.T)
        dim_ort = ort.shape[1]

        # forming an orthonormal basis using UCM and ORT vectors
        onb = np.concatenate((ort, ucm), axis = 1)

        return onb, dim_ort, dim_ucm


    # shared transform method across all ucm classes
    def transform(self, X, y = None):
        """
        y ignored
        """
        # check to see if model has been fit
        self._check_is_fitted()

        # check dimension match between jacobian and X
        if self.onb.shape[0] != X.shape[1]:
            raise(ValueError(f"Data matrix of different dimension ({X.shape[1]}) than orthonormal basis ({self.onb.shape[0]}). Use a different data matrix or Jacobian."))

        # continue to transform data
        self.projections = X @ self.onb

        # covariance of projections
        C_p = np.cov(self.projections.T)

        # collecting variances along ORT and UCM
        vort, vucm = np.split(np.diag(C_p), [self.dim_ort])

        # normalizing per dimension
        vort = sum(vort / self.dim_ort)
        vucm = sum(vucm / self.dim_ucm)

        # delta v
        dv = (vucm - vort)/(vucm + vort)

        # setting internal variables
        self.dv = dv
        self.vucm = vucm
        self.vort = vort

        return self

    
class UCMBayesRegress(_UCMmodel):
    """
    Bayesian model of the UCM.
    """
    pass


class UCMRegress(_UCMmodel):
    """
    Uses multiple linear regression to define the ucm.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        """
        Uses standard multiple regression between the elements X and
        the performance variable y to estimate the jacobian matrix.
        """
        # demean X
        X = X.copy() - np.mean(X, axis = 0)
        y = y.copy() - np.mean(y, axis = 0)
        
        # check to see if scaled data should be used
        if self._use_scaled_:
            X = self._scale_data(X)
            y = self._scale_data(y)
            
        # X and y have already been passed in the fit function
        # least-squares estimator of partial derivatives
        betas = linalg.pinv(X) @ y

        # setting jacobian
        self.j = betas

        # getting jacobian from the regression
        j_norm = self._check_j(betas)

        # getting results
        onb, dim_ort, dim_ucm = self._def_onb(j_norm)

        # score: R^2
        sse = sum(((X @ betas) - y)**2)
        sst = sum((y - np.mean(y))**2)
        score = 1 - (sse/sst)

        # changing _is_fitted_ internally
        self._is_fitted_ = True

        # reassignig computed values
        self.onb = onb
        self.dim_ort = dim_ort
        self.dim_ucm = dim_ucm
        self.score = score

        return self

    def fit_transform(self, X, y):
        """
        Fitting to data and producing synergy index results altogether at once
        """
        # fitting, then transforming.
        self.fit(X,y)
        self.transform(X)

        return self

    def _test(self, X, y = None):
        """
        EXPERIMENTAL HYPOTHESIS TEST FOR THE RESULTS
        """
        evls, evcs = np.eigh(np.cov(X.T))

        
class UCMAnalytical(_UCMmodel):
    """
    Scholz & Schoner 1999 (consider putting paper in assets)
    """

    def __init__(self, j):
        super().__init__()
        self._check_j(j)
        self.j = j
        
    def fit(self, X = None, y = None):
        """
        generates an orthonormal basis of the ucm and ort spaces.
        All arguments are ignored
        """
        # cheking and altering jacobian as needed
        j_norm = self._check_j(self.j)

        # getting onb, dim_ort and dim_ucm
        onb, dim_ort, dim_ucm = self._def_onb(j_norm)

        # score (no estimation)
        score = None
        
        # changing _is_fitted_ internally
        self._is_fitted_ = True

        # reassignig computed values
        self.onb = onb
        self.dim_ort = dim_ort
        self.dim_ucm = dim_ucm
        self.score = score

        return self

    def fit_transform(self, X, y = None):
        """
        Calls fit and transform sequentially.
        """
        self.fit()
        self.transform(X)
        return self


#X = np.array([[1,2,3],[7,2,2],[4,2,9],[4,4,8]])
#y = np.array([3,4,1,9])

#ucm_r = UCMRegress()
#ucm_r.fit_transform(X,y)
#pprint(ucm_r.__dict__)

#ucm_rr = UCMRegress()
#ucm_rr._use_scaled_ = True
#ucm_rr.fit_transform(X,y)
#pprint(ucm_rr.__dict__)

#j = np.array([[3,2,4]])
#ucm = UCMAnalytical(j)
#ucm.fit_transform(X)
# pprint(ucm.__dict__)




#################################################
############## Modes ############################
#################################################


class Modes(_Model):
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

    def fit(self, X, y = None):
        """
        y ignored.
        Will perform factor analysis using principal component method,
        with or without varimax rotation.

        The code for this function was adapted directly from factor_analyzer package
        """
        # perform principal component analysis on the data
        # simplified version of factor_analysis from factor_analyzer package
        # https://github.com/EducationalTestingService/factor_analyzer

        # and to get them after rotation, use commonalities (https://online.stat.psu.edu/stat505/lesson/12/12.11)

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


    def transform(self, X, y = None):
        """
        Ignores y input.
        """

        # check to see if model has been fit
        self._check_is_fitted()

        # check dimension match between jacobian and X
        if self.loadings.shape[0] != X.shape[1]:
            raise(ValueError(f"Data matrix of different dimension ({X.shape[1]}) than loadings ({self.loadings.shape[0]}). Use a different data matrix."))

        # using demeaned X
        X_dem = X - np.mean(X, axis = 0)

        # get projections
        self.magnitudes = X_dem @ self.loadings

        return self

    def fit_transform(self, X, y = None):
        """
        Ignores y input.
        """
        self.fit(X).transform(X)

        return self


# mm = Modes(rotation = "varimax")
# fin = UCMRegress().fit_transform(mm.fit_transform(X).magnitudes, y)
# pprint(fin.__dict__)
