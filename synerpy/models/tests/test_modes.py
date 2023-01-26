import pytest

from synerpy.datasets import load_forces
from synerpy.datasets import load_rates
from synerpy.models import Modes


forces = load_forces()
rates = load_rates()
M = Modes().fit(rates.data)
M.components_
