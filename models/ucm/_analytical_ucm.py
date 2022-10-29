from ._base import UCMModel
        
class AnalyticalUCM(UCMModel):
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
        
        # changing _is_fitted_ internally
        self._is_fitted_ = True

        # reassignig computed values
        self.onb = onb
        self.dim_ort = dim_ort
        self.dim_ucm = dim_ucm

        return self

    def fit_transform(self, X, y = None):
        """
        Calls fit and transform sequentially.
        """
        self.fit()
        self.transform(X)
        return self



