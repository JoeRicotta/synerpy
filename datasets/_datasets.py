from ._readers import force_read
from ._readers import delsys_read

def load_forces():
    force = force_read("synerpy/datasets/force_ex.txt")
    force.mvc_file = "mvc_ex.txt"
    force.mvc = 101.25
    return force

def load_firings():
    return delsys_read("synerpy/datasets/firings_ex.txt")

def load_rates():
    return delsys_read("synerpy/datasets/rates_ex.txt")

