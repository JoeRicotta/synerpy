import matplotlib.pyplot as plt
from statistics import mean, variance, quantiles, mode, geometric_mean
from factor_analyzer import FactorAnalyzer
from math import log
from scipy.stats import gamma, poisson
from collections.abc import Iterable
import numpy as np
import warnings


### TODO:
# 1. Include class for Bayesian interpretation of firings
# 2. Add documentation for all functions


def read(file):
    """
    Read a Delsys file and return the appropriate object based on the file format. This
    function will automatically interpret the file format and read the data, returning
    a file of the appropriate type. (Note: only supports Firings and Rates file inputs).
    
    Input:
        file: (str) the pathname of the desired file.

    Output:
        Firings object or Rates object

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
    
    # parsing between Firings and Rates file
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
        return Firings(file = file, data = data)
    
        
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
        data = np.array(data)
        return Rates(file = file, data = data)

    # if file is not recognized as rates or firings, throw error
    else:
        raise(ValueError("The file selected is not recognized as a Delsys Rates or Firings file."))



    
class Firings:
    """
    A class representing a spike train, or a series of action potentials.
    """

    def __init__(self, file, data):
        self.file = file
        self.data = data
        self.n_mus = len(self.data)
        self.time_range = (min(map(min, data)), max(map(max, data)))

    def __repr__(self):
        return f'Firings("{self.file}", time_range = {self.time_range})'
        
    def split(self, min_time, max_time):
        """
        Return a chunk of the dataset between time min_time and max_time
        """
        data = self.data.copy()
        
        # looping through each mu and getting only matching values within range
        out = list(map(lambda y: list(filter(lambda x: x > min_time and x < max_time, y)), data))
            
        # now making masked array, since every mu has different number of APs within range
        longest = max(map(len, out))
        for i,mu in enumerate(out):
            
            # add 0s for missing values
            dif = longest - len(mu)
            out[i] = out[i] + ([0] * dif)

        # coercing to masked array and returning as Firings object
        out = np.ma.masked_values(out, 0)

        return Firings(file = self.file, data = out)

    
class Rates:
    """
    A class built to handle instantaneous (i.e., continuous) firing rate data
    """

    def __init__(self, file, data):
        self.file = file
        self.data = data
        self.n_mus = self.data.shape[1] - 1
        self.time_range = (self.data[0][0], self.data[-1][0])

    def __repr__(self):
        return f'Rates("{self.file}", time_range = {self.time_range})'

    def __add__(self, other):
        """
        Joins time chunks together across the same set of motor units.
        Note: adding more than two chunks together in a non-sequential order
        might lead to strange behavior. Loop through the time chunks sequentially
        to avoid this.
        """
        data = np.concatenate((self.data, other.data), axis = 0)
        
        return Rates(file = f"{self.file[0:4]}... & {other.file[0:4]}...", data = data)

    def __radd__(self, other):
        """
        For using sum function across motor units
        """
        return self

    def __or__(self, other):
        """
        Joins motor units together across motor units, without regard for whether
        motor units between sets are identical.
        """
        data = []
        for a,b in zip(self.data, other.data):
            data.append(a.tolist() + b.tolist()[1:])
        return Rates(file = f"{self.file[0:4]}... & {other.file[0:4]}...", data = np.array(data))

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
        return Rates(f"{self.file[0:4]}... & {other.file[0:4]}...", data = data)

    
    def split(self, min_time, max_time):
        """
        Takes a subset of data within the specified time interval
        """

        # filtering old data
        data = np.array(list(filter(lambda x: min_time < x[0] <= max_time, self.data)))

        # returning what was found
        return Rates(file = self.file, data = data)
    
    
    def downsample(self, n_chunks):
        """
        Splits data into n_chunks chunks and averages within each chunk.
        """
        
        split = np.array_split(self.data, n_chunks)

        data = np.array([np.mean(x, axis = 0).tolist() for x in split if len(x) >= 1])

        # checking for lengths shorter than minimum resolution
        if len(data) < n_chunks:
            
            while len(data) < n_chunks:
                data = np.row_stack((data, data[-1]))
                # data.append(data[-1])
        
        return Rates(file = self.file, data = data)

    def modes(self, rotation = None, n_factors = 8, svd_method = "randomized"):
        """
        Generate motor unit modes using factor analysis.
        Note: requires the factor_analyzer package as a dependency

        rotation: "none", or "varimax", among others
        """

        # getting time from data
        time = self.data.T[0]
        
        # using factor analyzer to generate factor analysis object
        fa = FactorAnalyzer(n_factors, rotation = rotation, method = "principal", svd_method = svd_method)

        # fitting the model
        fa.fit(self.data.T[1:].T) # fitting data, excluding time column
        eigenvals, _ = fa.get_eigenvalues()
        eigenvals = eigenvals[0:n_factors] # since all eigenvalues are returned, only need a subset
        tot_var = self.n_mus # since the correlation matrix is used to determine the eigenvalues, this represents n_mus with variance 1 each.
        loadings = fa.loadings_

        # variance explained
        var_exp = eigenvals / tot_var

        # using chosen loadings to generate mode magnitudes (i.e.: the data of the Mode class.)
        data = self.data.T[1:].T @ loadings

        # same as data, but has unit variance and 0 mean. No need to use at this time,
        # but handy to have aroud for future variations...
        # magnitudes = fa.transform(self.data.T[1:].T)
        
        return Modes(file = self.file, data = data, time = time, loadings = loadings, eigenvalues = eigenvals, var_exp = var_exp)



class Modes:

    def __init__(self, file, data, time, loadings, eigenvalues, var_exp):

        # general attributes of the object
        self.file = file
        self.data = data
        self.time = time
        self.time_range = (time[0], time[-1])
        self.eigenvalues = eigenvalues
        self.loadings = loadings
        self.var_exp = var_exp
        self.n_modes = 0 if len(loadings.shape) == 1 else loadings.shape[1]

    def __repr__(self):
        return f"Modes({self.file}, time_range = {self.time_range}, shape = {self.loadings.shape})"

    def __iter__(self):
        # returns some iterable object
        out = [self.loadings.T, self.eigenvalues, self.var_exp, self.data.T]
        out = [x for x in zip(*out)]
        return iter(out)

    def filter(self, f):
        """
        Filters modes based on significance criteria with respect to some
        attribute of the mode class.

        Takes in a function and returns the object based on that function
        by iterating through each mode and testing it against criteria

        ex:
        Modes.filter(lambda x: x.eigenvalues > .1)
        """
        out = None
        bools = []
        for i, mode in enumerate(self):
            ith_mode = Modes(file = self.file, data = self.data.T[i], time = self.time,
                            loadings = self.loadings.T[i], eigenvalues = self.eigenvalues[i],
                            var_exp = self.var_exp[i])

            # adding whether to keep or remove the mode
            bools.append(f(ith_mode))

        return Modes(file = self.file, data = self.data, time = self.time, loadings = self.loadings.T[bools].T,
                     eigenvalues = self.eigenvalues[bools], var_exp = self.var_exp[bools])
                
