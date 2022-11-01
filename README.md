# Synerpy
A toolkit for the analysis of synergies in motor control and motor neuroscience

Synerpy extends scikit-learn's powerful tools of machine learning for use in motor neuroscience. You will need sklearn v1.1.3 or greater to use synerpy.

Broadly speaking, the word synergy is used to define the existence of significant covariation between elements of the motor ensemble, although the exact definition varies between authors. Synerpy, while still nascent, aims to offer a single resource for the use of the myriad analytical techniques used within the analysis of synergies.

## `synerpy.utils` example: motor unit modes, analysis of synergy
(Madarshahian et al. 2021)

Below is an example of using `synerpy.utils` to construct motor unit modes.
```python
from synerpy.utils import spikes
from synerpy.utils import force

# file directories (may need to change on your local device)
fr_fl = "/example_data/force_ex.txt"
mvc_fl = "/example_data/mvc_ex.txt"
rates_fl = "/example_data/rates_ex.txt"
```
The sample data was taken from an experiment involving one element producing a sine-wave profile of ankle dorsiflexion force between 20 and 40% MVC. We will use the force data to identify motor unit-modes by
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
force_tr = fr.read(fr_fl).normalize(mvc_fl).butterworth(10, hz).split(2,15)

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
mu_tr = spikes.read(rates_fl)

# segmenting and downsampling according to accepted force cycles
mu_seg = [x.split(t[0], t[1]).downsample(10) for t in acc_times]

### motor unit modes
# to get motor unit modes, concatenate the accepted cycles by using sum()
mu_cat = sum(mu_seg)

# to get MU-modes:
all_modes = mu_cat.modes(n_factors = 8, rotation = "varimax")

# use an anonymous function in .filter() to accept only those modes defined as significant
f = lambda x: x.eigenvalues > 1 and any(x.loadings) > .5
modes = mu_modes.filter(f)

# these are the motor-unit modes, complete with data and loading factors
print(modes.data)
print(modes.loadings)

```

# Why a single package?
1. Transparency. The use of a single, citable package allows for reproduction of results and is aligned with best scientific practice.
2. Ease. It is (theoretically) possible to recreate every analytical technique from the papers within which they are presented, just as it is theoretically possible to recreate the wheel time and time again.
3. Interpretability. A single repository offers a space for the community of motor control & motor neuroscience researchers to discuss analytical techniques, and for the code to evolve with the field.