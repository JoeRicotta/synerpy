import os
from collections.abc import Iterable
from scipy import signal
import numpy as np
import warnings
from matplotlib import pyplot as plt


class BaseTimeSeries(object):

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
                data = data.T[1:].T#[0].T

        else:
            # if only one column, clearly need a time index
            time = np.arange(max(data.shape))

        return data, time


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
        if type(self) == type(other):
            data = self.data + other.data
        else:
            data = self.data + other
        return data

    def __sub__(self, other):
        if type(self) == type(other):
            data = self.data - other.data
        else:
            data = self.data - other
        return data

    def __mul__(self, other):
        if type(self) == type(other):
            data = self.data * other.data
        else:
            data = self.data * other
        return data

    def __truediv__(self, other):
        if type(self) == type(other):
            data = self.data / other.data
        else:
            data = self.data / other
        return data

    def __matmul__(self, other):
        if type(self) == type(other):
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
        
