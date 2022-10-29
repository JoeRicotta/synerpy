from ._base import BaseTimeSeries

class Force(BaseTimeSeries):
    """
    A class to capture the generic nature of recorded force data
    (Note: the assumed units of time are in ms).
    """

    time_units = "ms"

    def __init__(self, file, data, time, units = "N", mvc_file = None, mvc = None):

        super(Force, self).__init__(file = file, data = data, time = time)
        self.mvc = mvc
        self.mvc_file = mvc_file
        self.time_range = (min(time) / 1000, max(time) / 1000)
        self.units = units
        
    @property
    def T(self):
        file, data, time = super().T
        return Force(file, data, time, mvc_file = self.mvc_file, mvc = self.mvc, units = self.units)

    def __add__(self, other):
        data = super().__add__(other)
        return Force(None, data = data, time = self.time, mvc_file = self.mvc_file, mvc = self.mvc, units = self.units)

    def __sub__(self, other):
        data = super().__sub__(other)
        return Force(None, data = data, time = self.time, mvc_file = self.mvc_file, mvc = self.mvc, units = self.units)
    
    def __mul__(self, other):
        data = super().__mul__(other)
        return Force(None, data = data, time = self.time, mvc_file = self.mvc_file, mvc = self.mvc, units = self.units)

    def __truediv__(self, other):
        data = super().__truediv__(other)
        return Force(None, data = data, time = self.time, mvc_file = self.mvc_file, mvc = self.mvc, units = self.units)
    
    def __repr__(self):
        return "Forces" + super().__repr__() + f", units = {self.units})"
    
    def __or__(self, other):
        """
        Analogous to union. Joins the datasets columnwise, and returns common time-- or if uncommon, a None time.
        """
        # check same units
        assert self.units == other.units, "Units do not match between force objects"
        
        data, time = super().__or__(other)
        
        return Force(None, data = data, time = time,  mvc_file = self.mvc_file, mvc = self.mvc, units = self.units)

    def append(self, other):
        """
        Joins two datasets rowwise, and tries to do so with time as well.
        """
        data, time = super().append(other)

        return Force(None, data = data, time = time,  mvc_file = self.mvc_file, mvc = self.mvc, units = self.units)
        
    
    def butterworth(self, cutoff_freq, sample_hz, order = 4, btype = "lowpass"):
        """
        butterworth filters the data using a digital butterworth, default 4th order lowpass.
        """
        # creating butterworth parameters
        sos = signal.butter(order, cutoff_freq, fs = sample_hz, output = "sos", btype = btype)
        data = signal.sosfilt(sos, self.data.T).T
        return Force(file = self.file, data = data, time = self.time, mvc_file = self.mvc_file, mvc = self.mvc, units = self.units)

    def segment(self, start_s, end_s):

        data, time = super().segment(start_s, end_s)
        
        return Force(self.file, data = data, time = time, mvc = self.mvc, mvc_file = self.mvc_file, units = self.units)


    def normalize(self, mvc_file = None):
        """
        Normalizes force data given an MVC file.
        """

        # check to see if mvc_file is passed as an argument;
        # if not, grab as attribute or raise error
        if not mvc_file:
            assert self.mvc_file, 'No MVC file declared'
            mvc_file = self.mvc_file # set the mvc_file attribute to be used, if it exists

        # read mvc file and use largest value as mvc value
        stream = open(mvc_file)
        st = 0
        for line in stream:
            val = float(line.rstrip())
            st = val if val > st else st
        mvc = st

        # normalize data and return
        data = self.data / mvc

        return Force(file = self.file, data = data, time = self.time, mvc = mvc, mvc_file = mvc_file, units = "%MVC")

    
    def downsample(self, n_chunks):
        """
        Downsample data to n_chunks in row length, along with time.
        """
        # calling super method
        data, time = super().downsample(n_chunks)

        return Force(file = self.file, data = data, time = time, mvc = self.mvc, mvc_file = self.mvc_file, units = self.units)

    # FIXME
    # needs to make sense for data with more than one element.
    def partition(self, hz, method = "midline", target = .3, prominence = .055, distance_s = .25):        
        """
        Partition force time series based on force magnitudes.

        Inputs:
            hz: the sampling frequency of the dataset. Note that downsampling will change this value.
            method: This function comes with five methods:
                midline: cuts data at target value crossings
                minmax: cuts data at peaks and valleys
                min: cuts data at valleys
                max: cuts data at peaks
                cycle: cuts data at full cycles
            target: the center of the cycle in % MVC
            prominence: passed to signal.find_peaks() to find the peaks of the rectified force data
            distance_s: the minimum distance at which two peaks should be separated. This helps avoid
                        identifying peaks that are too close to each other, such as if there was an accidental
                        increase in force right after an actual peak.

        Returns:
            stuff.
        """
        # firstly, is the data unidimensional?
        assert self.data.shape[1] == 1, "Data must be a column matrix to use partition()"
        data = self.data.T[0]

        # calculating minimum number of indices between peaks
        distance = hz * distance_s

        # init lists, demean force profile and ID which points are positive
        maxes, mins = [], []
        centered = data - target
        sign = centered > 0

        # find indices of peaks in rectified data
        inds = signal.find_peaks(np.abs(centered), prominence = prominence, distance = distance)[0]

        # sorting indices between positive and negative data
        for i in inds:
            maxes.append(i) if sign[i] else mins.append(i)

        # storing minmax as the union of all crossings
        minmax = mins + maxes
        minmax.sort()

        # storing a full cycle as every other index
        cycle = [j for i,j in enumerate(inds) if i % 2 == 0]

        # initializing dict of methods from which to grab the
        # prompted method
        methods = {"midline": inds,
                   "max": maxes,
                   "min": mins,
                   "minmax": minmax,
                   "cycle" : cycle}
      
        # getting the proper indices from the list
        cuts = methods.get(method)
      
        # grabbing from list
        out, time = [[]], [[]]
        j = 0
        for i, line in enumerate(zip(data, self.time)):
            out[j].append(line[0].tolist())
            time[j].append(line[1].tolist())
            if i in cuts:
                out.append([])
                time.append([])
                j += 1
               
        out = [np.array([x]).T for x in out]
        time = [np.array(x) for x in time]
        
        # yielding individual force objects
        for el in zip(out, time):
            yield Force(file=self.file, data=el[0], time=el[1], mvc_file=self.mvc_file, mvc=self.mvc, units=self.units)
