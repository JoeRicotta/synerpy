import pytest

from synerpy.datasets import load_forces
from synerpy.datasets import load_firings
from synerpy.datasets import load_rates

forces = load_forces()
firings = load_firings()
rates = load_rates()

dir(forces)
dir(firings)
dir(rates)
