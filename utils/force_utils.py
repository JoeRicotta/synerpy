""" helper functions with force data """

from scipy import signal
import numpy as np


# TODO: make this compatible with force files of multiple elements

def ensure_time_col(data):
    """
    checks to see if data reasonably has time column, and if not, creates a generic one

    Input:
        data (np.array)
    """

    # checking to see if a column is monotonically
    # increasing and saving it as time index; if
    # none exists, create generic one
    
    if len(data.shape) > 1:
        # if the ncols is greater than one, check to see if time column existsn
        diffed = np.diff(data.T[0])
        diffed = diffed[0:len(diffed) - 5] # chopping off the last few values for an error in our force recording
        has_time = all([x >= 0 for x in diffed])

        if has_time:
            # if time column is found, separate from the data
            time = data.T[0]
            data = data.T[1:][0]
            
    else:
        # if only one column, clearly need a time index
        time = np.arange(max(data.shape))

    return data, time


def read(file, delim = "\t"):
    """
    Reads in data as a force file
    """
    
    stream = open(file)
    data = []
    for i, line in enumerate(stream):
        split = line.rstrip().split(delim)
        split = [float(x) for x in split]
        data.append(split)
    data = np.array(data)

    # asserting that a time column exists, and if not, creating one
    data, time = ensure_time_col(data)

    # finally, return force object
    return Force(file = file, data = data, time = time, units = "N")

    

class Force:
    """
    A class to capture the generic nature of recorded force data
    (Note: the assumed units of time are in ms).
    """

    def __init__(self, file, data, time, units, mvc_file = None, mvc = None):
        self.file = file
        self.data = data
        self.mvc = mvc
        self.mvc_file = mvc_file
        self.time = time
        self.time_range = (min(time) / 1000, max(time) / 1000)
        self.units = units

    def __add__(self, other):
        """
        Appends together force data and time data between the trials
        """
        assert self.units == other.units, "Units do not match between force objects"

        data = np.append(self.data, other.data, axis = 0)
        time = np.append(self.time, other.time, axis = 0)
        return Force(self.file, data = data, time = time, mvc_file = self.mvc_file, mvc = self.mvc, units = self.units)

    def __radd__(self, other):
        """
        To use so that force objects can be used in a sum call.
        """
        return self

    def __repr__(self):
        return f'Forces("{self.file}", time_range = {self.time_range}, units = {self.units})'

    
    def butterworth(self, cutoff_freq, sample_hz, order = 4, btype = "lowpass"):
        """
        butterworth filters the data using a digital butterworth, default 4th order lowpass.
        """

        sos = signal.butter(order, cutoff_freq, fs = sample_hz, output = "sos", btype = btype)
        data = signal.sosfilt(sos, self.data)

        return Force(file = self.file, data = data, time = self.time, mvc_file = self.mvc_file, mvc = self.mvc, units = self.units)

    def split(self, start_s, end_s):
        """
        Take subset of data, based on time, and return as new force object. Note the
        time interval is a clopen set, i.e.,
            t ∈ (start_s, end_s]
        and includes the uppermost time while excluding the lower time.

        Input:
            start_s: the start time of the split chunk in seconds
            end_s: the end time of the split chunk in seconds
        """

        # converting time to ms for easier indexing
        start_s *= 1000
        end_s *= 1000

        # getting values of the force data which match this profile and returning new force object
        inds = [x and y for x,y in zip(self.time < end_s, self.time >= start_s)]
        
        return Force(self.file, data = self.data[inds].T, time = self.time[inds], mvc = self.mvc, mvc_file = self.mvc_file, units = self.units)


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
        Takes data and breaks into n_chunks, averaging across chunks
        """

        chunked_data = np.array_split(self.data, n_chunks, axis = 0)
        chunked_time = np.array_split(self.time, n_chunks, axis = 0)

        data, time = [], []
        for chunk in zip(chunked_data, chunked_time):
            data.append(chunk[0].T.mean(axis = 0).T)
            time.append(chunk[1].mean(axis = 0))
        data = np.array(data)
        time = np.array(time)
        
        return Force(file = self.file, data = data, time = time, mvc = self.mvc, mvc_file = self.mvc_file, units = self.units)

    
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

        # calculating minimum number of indices between peaks
        distance = hz * distance_s

        # init lists, demean force profile and ID which points are positive
        maxes, mins = [], []
        centered = self.data.T - target
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
        cycle = [i for i,j in enumerate(inds) if j % 2 == 0]

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
        for i,line in enumerate(zip(self.data, self.time)):
            out[j].append(line[0].tolist())
            time[j].append(line[1].tolist())
            if i in cuts:
                out.append([])
                time.append([])
                j += 1
                
        out = [np.array(x) for x in out]
        time = [np.array(x) for x in time]

        # yielding individual force objects
        forces = []
        for el in zip(out, time):
            yield Force(file = self.file, data = el[0], time = el[1], mvc_file = self.mvc_file, mvc = self.mvc, units = self.units)


# example
# # f = read('force_ex.txt')
# filt = f.butterworth(10,1000)
# norm = filt.normalize('mvc_ex.txt')
# spl = norm.split(2,15)
# out = list(spl.partition(hz = 1000))
# fin = list(map(lambda x: x.downsample(10), out))
