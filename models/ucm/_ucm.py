from sklearn.base import BaseEstimator

from ._base import _BaseUCM

class UCM(_BaseUCM, BaseEstimator):

    def fit(self, scores, jacobian):
        
        self.jacobian_ = self._check_j(jacobian)
        self.scores_ = self._validate_data(X = scores)

        # ensure compatible sizes for matrix product between
        # data and jacobian, since they're manually entered here
        if self.scores_.shape[1] != self.jacobian_.shape[1]:
            raise ValueError(
                f"Mismatch between scores dim1 ({self.scores_.shape[1]}) "
                f"and jacobian dim1 ({self.jacobian_.shape[1]})."
                )
        
        self._ucm_fit()

        return self
        


# X = np.random.rand(6,2)
# j = np.random.rand(2).reshape(1,2)

# M = UCM().fit(X,j)
