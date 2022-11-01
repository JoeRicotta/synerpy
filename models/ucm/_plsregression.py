from sklearn.cross_decomposition import PLSRegression

from ._base import _BaseUCM

# API to sklearn regression and sklearn PCA.
# Should be safe wrt version updates.

class _PLSRegression(PLSRegression, _BaseUCM):

    def fit(self, X, Y):
        
        super().fit(X,Y)

        # to adjust for strange coefficients coming
        # out of PLS regression
        if self.coef_.shape[0] > self.coef_.shape[1]:
            self.jacobian_ = self.coef_.T
        else:
            self.jacobian_ = self.coef_
            
        self.scores_ = self.x_scores_

        # fiting ucm
        self._ucm_fit()

        return self


