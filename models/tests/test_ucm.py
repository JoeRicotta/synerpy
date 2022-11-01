import numpy as np

from synerpy.models import ucm

# defining elements
elements = np.random.rand(6,3)

# standard ucm: when the jacobian is known a-priori
standard_ucm = ucm.UCM()
jacobian = np.array([[1,1,1]]) # notice this is 2-dimensional!
standard_ucm.fit(elements, jacobian)
standard_ucm.synergy_index_ # result: synergy index
standard_ucm.ucm_ # result: orthonormal basis for the ucm


###
# Empirical UCM: when the jacobian linking elements to
# performance needs to be estimated.
performance = np.random.rand(6)

# pc regression
pcreg = ucm.PCRegression()
pcreg.fit(elements, performance)
pcreg.jacobian_
pcreg.synergy_index_

# pls regression
plsreg = ucm.PLSRegression() # throws an ugly but unmutable warning.
plsreg.fit(elements, performance)
plsreg.jacobian_
plsreg.synergy_index_

# standard regression
reg = ucm.LeastSquaresRegression()
reg.fit(elements, performance)
reg.jacobian_
reg.synergy_index_

