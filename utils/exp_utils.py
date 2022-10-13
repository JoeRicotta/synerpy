"""
Experimental utilities useful for
    1. Identifing files in a given directory using keyword IDs, and
    2. Setting experiment-wide paremeters.
"""

import os
from collections.abc import Iterable

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
