"""Functions for computation of the partial derivative of atmosphere state parameters."""
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

from functools import lru_cache, singledispatch
from numbers import Number

import numpy as np
from scipy.constants import R, atmosphere

AIR_MOLAR_MASS = 28.9647e-3
AIR_GAS_CONSTANT = R / AIR_MOLAR_MASS
SEA_LEVEL_PRESSURE = atmosphere
SEA_LEVEL_TEMPERATURE = 288.15
TROPOPAUSE = 11000.0


# PARTIAL TEMPERATURE =================================================================
@singledispatch
def compute_partial_temperature(altitude, unit_coeff) -> np.ndarray:
    """
    :param altitude: in m
    :param unit_coeff: coefficient to adjust the partial computation if the altitude is in feet

    :return: Partial of temperature in K with respect to altitude
    """
    # Implementation for numpy arrays
    idx_tropo = altitude < TROPOPAUSE

    # Since the derivative for the stratosphere is zero, we will start with a zeros_like and not
    # an empty_like
    partial_temperature = np.zeros_like(altitude)
    partial_temperature[idx_tropo] = -0.0065 * unit_coeff

    return partial_temperature


@compute_partial_temperature.register
@lru_cache()
def _(altitude: Number, unit_coeff) -> float:
    # Implementation for floats
    if altitude < TROPOPAUSE:
        partial_temperature = -0.0065 * unit_coeff
    else:
        partial_temperature = 0.0
    return partial_temperature
