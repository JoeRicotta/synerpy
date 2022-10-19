"""
Utilities broken into
1. Experimental utilities,
2. Force utilities
3. Neuro utilities
"""
import os
from collections.abc import Iterable
from scipy import signal
import numpy as np
import warnings
from matplotlib import pyplot as plt


### TODO:
# documentation for all of the below (sheesh...)
# getter / setter for data transposes

### LAST LEFT OFF AT:
# Moving as many methods to trial as possible.
# Need to test and reconcile Spikes and Rates class to this.

####################################################
################### base classes  ##################
####################################################

class _Trial(object):

    def __init__(self, file, data, time = None):
        
        # input checks
        time_range = self._time_check(time)
        self._data_check(data)
        self._time_data_check(time, data)

        # if all pass scrutiny, create object with attributes
        self.file = file
        self.data = data
        self.time = time
        self.time_range = time_range

    # need to think about how these will interact in subclasses
    @property
    def T(self):
        # this is a "getter" for property T.
        # this will only run when _Trial(...).T is called.
        # used in subclasses to transpose data
        return self.file, self.data.T, self.time

    @staticmethod
    def _time_check(time):
        """
        If time is passed, make sure it is a numpy array or masked array.
        """
        if time is not None:
            assert isinstance(time, np.ndarray) or isinstance(time, np.ma.core.MaskedArray), "Time vector must be numpy array."

            # output time range
            return (time.min(), time.max())

    @staticmethod
    def _data_check(data):
        """
        Make sure type of data is correct and dimension of data is correct
        """
        assert isinstance(data, np.ndarray), "Data must be an np.array (masked or unmasked)"
        assert len(data.shape) == 2, "Data matrix must be two-dimensional numpy array."

    @staticmethod
    def _time_data_check(time, data):
        """
        If time and data are passed, check to make sure time array is 1-d and that it shares a length with data.
        """
        if time is not None:
            assert len(time.shape) == 1, f"Expected time array of dimension 1 but recieved time array of dimension {len(time.shape)}."
            assert time.shape[0] in data.shape, f"Time and data arrays share no common length; time is length {time.shape[0]} and data of shape {data.shape}."

    def __repr__(self):
        return f"('{self.file}', shape={self.data.shape}, time_range={self.time_range}"

    def __add__(self, other):
        if type(other) == type(Trial):
            data = self.data + other.data
        else:
            data = self.data + other
        return data

    def __sub__(self, other):
        if type(other) == type(Trial):
            data = self.data - other.data
        else:
            data = self.data - other
        return data

    def __mul__(self, other):
        if type(other) == type(Trial):
            data = self.data * other.data
        else:
            data = self.data * other
        return data

    def __truediv__(self, other):
        if type(other) == type(Trial):
            data = self.data / other.data
        else:
            data = self.data / other
        return data

    def __matmul__(self, other):
        if type(other) == type(Trial):
            data = self.data @ other.data
        else:
            data = self.data @ other
        return data

    def __or__(self, other):
        """
        Analogous to union. Joins the datasets columnwise, and returns common time-- or if uncommon, a None time.
        """
        # joining data columnwise
        data = np.concatenate((self.data, other.data), axis = 1)
        if self.time is not None and other.time is not None and all(self.time == other.time):
            time = self.time
        else:
            time = None
        return data, time

    def __radd__(self,other):
        return self

    def append(self, other):
        """
        Joins two datasets rowwise, and tries to do so with time as well.
        """
        # joining data rowwise
        data = np.concatenate((self.data, other.data), axis = 0)

        # working with time now
        if self.time is not None and other.time is not None:
            time = np.concatenate((self.time, other.time), axis = 0)
        else:
            time = None

        return data, time

    def plot(self):
        """
        creates and shows a plot
        """
        plt.figure()
        plt.title(self.__repr__())
        plt.xlabel("Time (s)")
        if self.time is not None:
            if self.time_units == "ms":
                time = self.time / 1000
            else:
                time = self.time
            plt.plot(time, self.data)
        else:
            plt.plot(self.data)
        plt.show()

        return self

    def downsample(self, n_chunks):
        """
        Takes data and breaks into n_chunks, averaging across chunks
        Will exclude any cuts that have < n_chunks datapoints
        """
        # handling missing data: if n_chunks exceeds the data chunk length,
        # simply return the original data here
        if n_chunks >= self.data.shape[0]:
            return self.data, self.time

        # otherwise, continue with the analysis
        chunked_data = np.array_split(self.data, n_chunks, axis = 0)
        chunked_time = np.array_split(self.time, n_chunks, axis = 0)

        # averaging across chunks
        aver_data = [np.array([x.tolist()]) for x in map(lambda x: x.mean(axis = 0), chunked_data)]
        data = np.concatenate(aver_data, axis = 0)

        # checking the same for time
        if self.time is not None:
            time = np.concatenate([np.array([x.tolist()]) for x in map(lambda x: x.mean(axis = 0), chunked_time)], axis = 0)
        else:
            time = None

        return data, time

    def segment(self, start_s, end_s):
        """
        Take subset of data, based on time, and return as new force object. Note the
        time interval is a clopen set, i.e.,
            t ∈ (start_s, end_s]
        and includes the uppermost time while excluding the lower time.

        Input:
            start_s: the start time of the segment chunk in seconds
            end_s: the end time of the segment chunk in seconds
        """
        if self.time_units == "ms":
            start_s *= 1000
            end_s *= 1000

        # getting indices of the trial in this window
        inds = (self.time > start_s) & (self.time < end_s)

        return self.data[inds], self.time[inds]
        



####################################################
################### Experimental utils #############
####################################################


# Parameters
PARAMS = {}

# variable with data directory
DATA_DIR = "utils/example_data/"


class ID:
    """
    A class meant to help use search criteria to identify files
    """

    def __init__(self, ID, search_dir = os.getcwd(), **kwargs):
        self.ID = ID
        files = []
        for root, __, file_ in os.walk(search_dir):
            files += [root + f if root[-1] == "/" else root + "/" + f for f in file_]
        files_ = kwargs.get("files") or [x for x in files if ID in x]
        files_.sort()
        self.files = files_

    def __mul__(self, other):
        return ID(ID = f"({self.ID} & {other.ID})", files = list(set(self.files).intersection(other.files)))

    def __add__(self, other):
        return ID(ID = f"({self.ID} | {other.ID})", files = list(set(self.files).union(other.files)))

    def __and__(self, other):
        return self.__mul__(other)

    def __or__(self, other):
        return self.__add__(other)

    def __iter__(self):
        return iter(self.files)

    def __contains__(self, item):
        return item in self.files

    def __repr__(self):
        return self.ID

    def __getitem__(self, key):
        return self.files[key]


class Subject(ID):
    """
    Subject class (inherits ID class)

    """
    def __init__(self, subj_ID, search_dir = os.getcwd(), prefix = "subj"):
        super().__init__(prefix + str(subj_ID), search_dir)
        self.subj_ID = subj_ID
        self.prefix = prefix
        

class Trial(ID):
    """
    Trial class (inherits ID class)

    """
    def __init__(self, trial_ID, search_dir = os.getcwd(), prefix = "t"):
        super(Trial, self).__init__(prefix + str(trial_ID), search_dir)
        self.trial_ID = trial_ID
        self.prefix = prefix


class Sensor(ID):
    """
    Sensor class (inherits ID class)
    """
    def __init__(self, sensor_ID, search_dir = os.getcwd(), prefix = "Sensor "):
        super(Sensor, self).__init__(prefix + str(sensor_ID), search_dir)
        self.sensor_ID = sensor_ID
        self.prefix = prefix


####################################################
################### Error utils ####################
####################################################

def exc_file_exists(file = '_excluded.txt'):
    files = os.listdir()
    return any([file in x for x in files])

def make_exc_file(file = '_excluded.txt'):
    if not exc_file_exists(file):
        stream = open(file, 'w')
        stream.close()
        print(f"Created file {file}")

def is_excluded(*args, file = '_excluded.txt'):
    """
    Checks to see if the specific information is in
    the file tracking exclusions.
    """
    # checking to see if file exists, or doing nothing if found
    make_exc_file(file)
    
    # setting criteria
    crit = list(args)
    crit = [str(x) for x in crit]

    # empty string being appended to search file for
    # search criteria
    emp = str()
    for c in crit:
        emp += c + "\t"

    # conducting search and return whether or not the search was found
    stream = open(file, 'r')
    already = any([emp in line for line in stream])
    stream.close()
    return already


def exclude(*args, file = '_excluded.txt'):
    """
    Places criteria in the exclusion file, if not already there.
    """
    if not is_excluded(*args, file = file):
        crit = list(args)
        stream = open(file, 'a')
        for c in crit:
            stream.write(str(c) + "\t")
        stream.write("\n")
        stream.close()
        print(f"Wrote line to file: \n\t{args}")

        
def clear_exclusions(file = "_excluded.txt"):
    """
    Clears the exclusion file
    """
    stream = open(file, "w")
    stream.close()



    

####################################################
################### Force utils ####################
####################################################

# TODO: build a more intuitive Reader class to import delimited files.

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


def force_read(file, delim = "\t"):
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

    # fixing dims
    if len(data.shape) == 1:
        data = np.array([data.tolist()]).T
        
    # finally, return force object
    return Force(file = file, data = data, time = time, units = "N")

    

class Force(_Trial):
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
# spl = norm.segment(2,15)
# out = list(spl.partition(hz = 1000))
# fin = list(map(lambda x: x.downsample(10), out))




####################################################
################### Neuro utils ####################
####################################################

def delsys_read(file):
    """
    Read a Delsys file and return the appropriate object based on the file format. This
    function will automatically interpret the file format and read the data, returning
    a file of the appropriate type. (Note: only supports Spikes and Rates file inputs).
    
    Input:
        file: (str) the pathname of the desired file.

    Output:
        Spikes object or Rates object

    Example:
        import mu_utils

        # for rates
        rates = mu_utils.read('rates_ex.txt')
        type(rates)

        # for firings
        firings = mu_utils.read('firings_ex.txt')
        type(firings)
       
    """
    
    # initializing data
    data = []

    # read file
    first_line = open(file).readline()
    stream = open(file)
    
    # parsing between Spikes and Rates file
    if 'Firings' in first_line:
        
        # if firings file, read data as firings data
        for i,line in enumerate(stream):
            split = line.rstrip().split('\t') # format line to be readable
            if i == 0:
                head = split
            else:
                conv = [round(float(x), 4) for x in split if x != ''] # append the nth firing if available
                while len(conv) < len(head): {conv.append(0)} # add nones to list for no firings
                data.append(conv) # add to data list

        # converting to ragged np array and returning
        # (note: since each motor unit has a unique number of firings, vector lengths are uneven
        # and data needs to be masked (i.e., filled with nans)).
        per_mu = [x for x in zip(*data)] # transposing to reflect individual motor units
        data = np.ma.masked_values(per_mu, 0) # making masked np array
        return Spikes(file = file, data = data)
    
        
    # otherwise, a rates file
    elif 'Time' in first_line:
        for i,line in enumerate(stream):
            split = line.rstrip().split('\t')
            if i == 0:
                head = split
            else:
                conv = [round(float(x), 4) for x in split if x != "" ] # grabbing data row-wise
                data.append(conv) # appending data to file

        # formatting data into rectangular np array and returning rates object
        time = np.array(data).T[0]
        data = np.array(data).T[1:].T
        return Rates(file = file, data = data, time = time)

    # if file is not recognized as rates or firings, throw error
    else:
        raise(ValueError("The file selected is not recognized as a Delsys Rates or Spikes file."))


    
class Rates(_Trial):
    """
    A class built to handle instantaneous (i.e., continuous) firing rate data
    """
    
    time_units = "s"

    def __init__(self, file, data, time = None):
        super(Rates, self).__init__(file = file, data = data, time = time)
        self.n_mus = self.data.shape[1]

    @property
    def T(self):
        file, data, time = super().T
        return Rates(file, data, time)

    def __add__(self, other):
        data = super().__add__(other)
        return Rates(None, data = data, time = self.time)

    def __sub__(self, other):
        data = super().__sub__(other)
        return Rates(None, data = data, time = self.time)
    
    def __mul__(self, other):
        data = super().__mul__(other)
        return Rates(None, data = data, time = self.time)

    def __truediv__(self, other):
        data = super().__truediv__(other)
        return Rates(None, data = data, time = self.time)

    def __repr__(self):
        return "Rates" + super().__repr__() + ")"

    def append(self, other):
        """
        Joins two datasets rowwise, and tries to do so with time as well.
        """
        data, time = super().append(other)
        return Rates(None, data, time)

    def __or__(self, other):
        """
        Analogous to union. Joins the datasets columnwise, and returns common time-- or if uncommon, a None time.
        """        
        data, time = super().__or__(other)
        
        return Rates(None, data = data, time = time)

    def __xor__(self, other):
        """
        Joins motor units together across motor units, while accounting for
        possible aliasing between sets of motor units.
        The set of unique MUs as determined by a correlation less than .96 between all
        MUs; averages frequency data between two MUs when aliasing occurs.
        """
        
        thresh = .96
        added = self | other

        time = added.data.T[0].tolist()
        data = added.data.T[1:]

        # getting correlation matrix from joined data
        # and prepping data to be edited
        corr = np.corrcoef(data)
        
        to_del = []
        for i,a in enumerate(corr):
            for j,b in enumerate(a):
                if i <= j or i < self.n_mus or j < other.n_mus or b < thresh:
                    pass
                else:
                    # average the two motor unit data to comprise a new motor unit and remove the former
                    data[i] = (data[i] + data[j]) / 2
                    to_del.append(j)

        # deleting all indices
        to_del = list(set(to_del))
        to_del.sort(reverse = True)
        for ind in to_del:
            data = np.delete(data, ind, 0)
            
        data = data.tolist()
        data.insert(0, time)
        data = np.array(data).T

        # returning object
        return Rates(f"{self.file[0:4]}... & {other.file[0:4]}...", data = data, time = self.time)


    def segment(self, start_s, end_s):
       
        data, time = super().segment(start_s, end_s)

        return Rates(self.file, data, time)
   
   
    def downsample(self, n_chunks):
        """
        Downsample data to n_chunks in row length, along with time.
        """
        # calling super method
        data, time = super().downsample(n_chunks)

        return Rates(self.file, data, time)





   
# It is not clear that Spikes should inherit the Trial class.
# Maybe an event class would be best fit for this?
class Spikes(_Trial):
    """
    A class representing a spike train, or a series of action potentials.
    """

    def __init__(self, file, data):
        super(Spikes, self).__init__(file = file, data = data)
        self.n_mus = len(self.data)
        self.time_range = (min(map(min, data)), max(map(max, data)))

    def __repr__(self):
        # return f'Spikes("{self.file}", time_range = {self.time_range})'
        return "Spikes" + super().__repr__() + ")"
       
    def segment(self, min_time, max_time):
        """
        Return a chunk of the dataset between time min_time and max_time
        """
        data = self.data.copy()
       
        # looping through each mu and getting only matching values within range
        out = list(map(lambda y: list(filter(lambda x: x > min_time and x < max_time, y)), data))
           
        # now making masked array, since every mu has different number of APs within range
        longest = max(map(len, out))
        for i, mu in enumerate(out):
           
            # add 0s for missing values
            dif = longest - len(mu)
            out[i] = out[i] + ([0] * dif)

        # coercing to masked array and returning as Spikes object
        out = np.ma.masked_values(out, 0)

        return Spikes(file = self.file, data = out)


# Trying out basic features
# ffile = "example_data/force_ex.txt"
# ff = force_read(ffile)
# ff.plot()
# ff.append(ff).plot()

mfile = "example_data/rates_ex.txt"
mm = delsys_read(mfile)
mm.segment(2,2.2).plot().downsample(20).plot()







