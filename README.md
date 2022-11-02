# `synerpy`
An open-source toolkit for the analysis of synergies in motor control and motor neuroscience.

Broadly speaking, the word synergy is used to define the existence of significant covariation between elements of the motor ensemble, although the exact definition varies between authors. `synerpy` offers a small but useful suite of extendable classes in the [sklearn](https://scikit-learn.org/stable/getting_started.html) 'call-fit-transform' style, permitting for the analysis of synergies. The current version hosts a suite of tools for the Uncontrolled Manifold Analysis (Scholz * Schoner 1999).

Synerpy uses scikit-learn's api to gently extend the powerful sklearn architecture with a motor neuroscience bent. You will need sklearn v1.1.3 or greater to use synerpy.

## Uncontrolled Manifold Analysis
The uncontrolled manifold analysis ([Scholz and Schoner 1999](https://www.researchgate.net/publication/12915817_Scholz_JP_Schoner_G_The_uncontrolled_manifold_concept_identifying_control_variables_for_a_functional_task_Exp_Brain_Res_126_289-306)) is a particular flavor of synergic analysis interested in understanding the geometric properties of variance in a basket of motor elements with respect to some given task.

 `synerpy.models.ucm` has two kinds of models, and the model one should use depends upon the jacobian matrix linking (infinitesimally) small changes in elemental variables to (infinitesimally) small changes in performance. Specifically, `synerpy.models.ucm` hosts:
1. an analytic model (`UCM()`), used when the jacobian matrix between elements and performance is known a-priori (i.e., before collecting data), and
2. empirical models (`PCRegression()`, `PLSRegression()`, `LeastSquaresRegression()`) used when the jacobian matrix is unknown and must be estimated from data.

### Analytic model
If the jacobian matrix was known before data was collected, the UCM analysis proceeds very simply. For the moment, we will simulate a small random dataset (since it does not make sense to perform this analysis on the loaded force/motor unit data).

```python
import numpy as np
from synerpy.models.ucm import UCM # importing analytic UCM model

# randomly generating dataset of 4 elements across 12 observations
elements = np.random.rand(12,4)
jacobian = np.array([[1,1,1,1]]) # a common jacobian for linear sums
```

Now we can load and fit the analytic UCM model.
```python
# loading the analytical UCM model
standard_ucm = UCM()

# fitting the model
standard_ucm.fit(elements, jacobian)

# observing the results
standard_ucm.synergy_index_
standard_ucm.vucm_

# hint: attributes ending in a "_", such as
# "standard_ucm.vort_"
# are parameters generated from fitting the model with .fit()
# This holds true for all of sklearn, and by proxy, synerpy.
```

### Empirical Model
An empirical UCM model might be encountered when an element of randomness or a latent relationship exists between the elemental and performance variables. For example, we might be interested in how the frequencies of firing of motor units may covary to stabilize force production within the tibialis anterior muscle during a cyclical force production task. Let's begin by loading some example force data and motor unit firing data within synerpy.

```python
from synerpy.datasets import load_forces # force example
from synerpy.datasets import load_rates # motor unit firing rates example

forces = load_forces()
rates = load_rates()
```

These return Force and Rates objects which exist within `synerpy.containers`, each with a host of useful data-specific processing features. (Hint: try `forces.plot()` or `rates.plot()`!)

Once these data are loaded, we can begin asking questions about them. `synerpy.models.ucm` comes equipped with three methods by which to define the jacobian, divided between whether i.) dimensionality reduction is needed, and ii.) model fitting is supervised or unsupervised. Below we see an example of all three:  

```python
################
# No dimensionality reduction
# Nomal ordinary least squares regression
#
# (PS: this is usually a bad idea if your elements
#  are highly correlated-- such as motor unit data!)

from synerpy.models.ucm import LeastSquaresRegression

# load the model
lsr = LeastSquaresRegression()

# fit the model
lsr.fit(rates.data, forces.data)

# check the results
lsr.jacobian_
lsr.synergy_index_


################
# Dimensionality reduction, unsupervised:
# Principal Component Regression
#
# Good for tackling multicolinearity and
# reducing dimensionality under the assumption
# elements are defined independently from
# the performance variable
#
# links:
# https://en.wikipedia.org/wiki/Principal_component_regression
# https://www.youtube.com/watch?v=MhB0G7Nb4fU

from synerpy.models.ucm import PCRegression

# load the model
pcr = PCRegression()

# fit the model
pcr.fit(rates.data, forces.data)

# check the results
pcr.jacobian_
pcr.synergy_index_


################
# Dimensionality reduction, supervised:
# Partial Least Squares Regression
#
# Good for tackling multicolinearity and
# reducing dimensionality under the assumption
# elements are defined independently from
# the performance variable
#
# links:
# https://en.wikipedia.org/wiki/Partial_least_squares_regression
# https://www.youtube.com/watch?v=Px2otK2nZ1c

from synerpy.models.ucm import PLSRegression

# load the model
pls = PLSRegression()

# fit the model
pls.fit(rates.data, forces.data)

# check the results
pls.jacobian_
pls.synergy_index_
```
## Containers for data

`synerpy` hosts a few helpful features for the segmenting and processing of common time series data. We have loaded examples of these before, but we are also able to load .txt files directly using read functions.

```python
from synerpy.datasets import force_read
from synerpy.datasets import delsys_read
from synerpy.models import Modes

# file directories (may need to change on your local device)
fr_fl = "/example_data/force_ex.txt"
mvc_fl = "/example_data/mvc_ex.txt"
rates_fl = "/example_data/rates_ex.txt"
```
We will use the force data to identify [motor-unit modes](https://pubmed.ncbi.nlm.nih.gov/36244637/) by
1. processing the force data (filter and normalize),
2. segmenting the up and down cycles,
3. downsampling them to a fixed number of datapoints,
4. removing cycles outside of a predetermined range, and then
5. using the times of the increases and decreases to segment the firing rates data from the motor units of the Tibialis Anterior muscle.

```python
### force processing
hz = 1000 # sampling frequency of force data
min_len, max_len = .3, .7 # minimum and maximum acceptable cycle length

# one-liner: import force data, normalize, low-pass filter, and segment from 2-15 s
force_tr = force_read(fr_fl).normalize(mvc_fl).butterworth(10, hz).split(2,15)

# partition based on force min/max, then downsample to 10 samples per partition
force_pr = force_tr.partition(hz, "minmax", .3, .015, .3) # outputs a generator of partitions

# using list comprehension to downsample each partition to 10 units
force_dn = [x.downsample(10) for x in force_pr]

# rejecting cycles based on the length of the partition
force_acc = [x for x in force_dn if min_len < x.time_range[1] - x.time_range[0] < max_len]

# obtaining accepted time slices
acc_times = [x.time_range for x in force_acc]

### motor unit processing
# reading in initial file
mu_tr = delsys_read(rates_fl)

# segmenting and downsampling according to accepted force cycles
mu_seg = [x.split(t[0], t[1]).downsample(10) for t in acc_times]

# documentation ends here--more to come soon!...

```

# Why a single package?
1. Transparency. The use of a single, citable package allows for reproduction of results and is aligned with best scientific practice.
2. Ease. It is (theoretically) possible to recreate every analytical technique from the papers within which they are presented, just as it is theoretically possible to recreate the wheel time and time again.
3. Interpretability. A single repository offers a space for the community of motor control & motor neuroscience researchers to discuss analytical techniques, and for the code to evolve with the field.

# Cite `synerpy`!
If you use synerpy in your work, please cite it ("python package synerpy v0.0.2 was used to perform uncontrolled manifold analysis", etc.) to help spread the open source tools!
