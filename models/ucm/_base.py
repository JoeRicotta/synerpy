from .._base import BaseModel

class UCMModel(BaseModel):
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


    # this used to be in transform but now this does not exist for this class.
    # it is not a transformer.
    """
    y ignored
    """
    # check to see if model has been fit
    _check_is_fitted(self)

    # check dimension match between jacobian and X
    if self.onb.shape[0] != X.shape[1]:
        raise(ValueError(f"Data matrix of different dimension ({X.shape[1]}) than orthonormal basis ({self.onb.shape[0]}). Use a different data matrix or Jacobian."))

    # continue to transform data
    self.projections = X @ self.onb

    # covariance of projections
    C_p = np.cov(self.projections.T)

    # collecting variances along ORT and UCM
    vort, vucm = np.split(np.diag(C_p), [self.dim_ort])

