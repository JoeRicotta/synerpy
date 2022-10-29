from ._base import UCMModel

class EmpiricalUCM(UCMModel):
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
