import numpy as np

import os

from ..containers._force_series import Force
from ..containers._rates_series import Rates
from ..containers._spike_train import Spikes


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
    data, time = Force.ensure_time_col(data)

    # fixing dims
    if len(data.shape) == 1:
        data = np.array([data.tolist()]).T
        
    # finally, return force object
    return Force(file = file, data = data, time = time, units = "N")



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



####################################################
################### Experimental utils #############
####################################################


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
