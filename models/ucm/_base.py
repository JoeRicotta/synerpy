import numpy as np
from scipy import linalg
import warnings

from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted




# Calls to UCM should be as follows:
# UCM(j). If j defined, straight to UCMAnalytical.
# if only X: UCM___? Need to develop. Probably Bayesian.
# if X, y: UCMEmpirical
#   3 submethods: UCMLSRegression, UCMPCRegression, UCMPLSRegression
# if X, J: UCMAnalytical


class _BaseUCM:
    """
    Base class for UCM.
    """
        
    @staticmethod
    def _check_j(j):
        """
        checks features of the jacobian and fixes them when possible.
        The jacobian matrix should be a wide matrix, or a square matrix.
        A long jacobian matrix means that the system is probably overdetermined, and the solution
        (if it exists) is singular.
        """
        j = check_array(j, ensure_2d=True)
        
        # making sure jacobian is properly shaped
        # as a wide or square matrix
        n_rows, n_cols = j.shape
        if n_rows > n_cols:
            warnings.warn(
                f"Jacobian matrix is long, with {n_rows} rows > {n_cols} columns. "
                "Expect weird behavior from an overdetermined system."
            )

        return j

    def _ucm_fit(self):
        """
        Takes a jacobian and defines an orthonormal basis of ORT and UCM.
        Returns the orthonormal basis as well as the dimensions of ORT and UCM.
        """
        
        # making sure j is normalized
        norm = np.diag((1 / linalg.norm(self.jacobian_, axis = 1)))
        normalized_jacobian = norm @ self.jacobian_

        # getting basis of nullspace of the jacobian matrix (ucm) and dimension
        ucm = linalg.null_space(normalized_jacobian)
        dim_ucm = ucm.shape[1]

        # getting ort from ucm and dimension
        ort = linalg.null_space(ucm.T)
        dim_ort = ort.shape[1]

        # forming an orthonormal basis using UCM and ORT vectors
        onb = np.concatenate((ort, ucm), axis = 1)

        # now making projections onto orthonormal basis
        cov_x = np.cov((self.scores_ @ onb).T)
        vort = np.diag(cov_x)[:dim_ort].sum()
        vucm = np.diag(cov_x).sum() - vort
        dv = vucm - vort / (vucm + vort)

        # storing all values
        self.ucm_ = ucm
        self.ort_ = ort
        self.dim_ucm_ = dim_ucm
        self.dim_ort_ = dim_ort
        self.onb_ = onb
        self.vucm_ = vucm
        self.vort_ = vort
        self.synergy_index_ = dv

        return self

