"""Functions for computation of atmosphere speed parameters."""
#  This file is part of StdAtm
#  Copyright (C) 2023 ONERA & ISAE-SUPAERO
#  StdAtm is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math
from functools import lru_cache, singledispatch
from numbers import Number

import numpy as np


#  This file is part of StdAtm
#  Copyright (C) 2023 ONERA & ISAE-SUPAERO
#  StdAtm is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


# TRUE AIRSPEED =================================================================
# From EAS =================================================================
@singledispatch
def compute_tas_from_eas(equivalent_airspeed, density, sea_level_density) -> np.ndarray:
    """

    :param equivalent_airspeed: in m/s
    :param density: in kg/m**3
    :param sea_level_density: in kg/m**3
    :return: true airspeed in m/s
    """
    # Implementation for numpy arrays
    true_airspeed = equivalent_airspeed * np.sqrt(sea_level_density / density)
    return true_airspeed


@compute_tas_from_eas.register
@lru_cache()
def _(equivalent_airspeed: Number, density: Number, sea_level_density: Number) -> float:
    # Implementation for floats
    true_airspeed = equivalent_airspeed * math.sqrt(sea_level_density / density)
    return true_airspeed


# From dynamic pressure =================================================================
@singledispatch
def compute_tas_from_pdyn(dynamic_pressure, pressure, speed_of_sound) -> np.ndarray:
    # Implementation for numpy arrays
    true_airspeed = np.sqrt(dynamic_pressure / 0.7 / pressure) * speed_of_sound
    return true_airspeed


@compute_tas_from_pdyn.register
@lru_cache()
def _(dynamic_pressure: Number, pressure: Number, speed_of_sound: Number) -> float:
    # Implementation for floats
    true_airspeed = math.sqrt(dynamic_pressure / 0.7 / pressure) * speed_of_sound
    return true_airspeed
