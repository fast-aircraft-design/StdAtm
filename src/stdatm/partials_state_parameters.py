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
from typing import Union

import math
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
    partial_temperature[idx_tropo] = -0.0065

    return partial_temperature * unit_coeff


@compute_partial_temperature.register
@lru_cache()
def _(altitude: Number, unit_coeff) -> float:
    # Implementation for floats
    if altitude < TROPOPAUSE:
        partial_temperature = -0.0065
    else:
        partial_temperature = 0.0
    return partial_temperature * unit_coeff


# PARTIAL PRESSURE =================================================================
@singledispatch
def compute_partial_pressure(altitude, unit_coeff) -> np.ndarray:
    """
    :param altitude: in m
    :param unit_coeff: coefficient to adjust the partial computation if the altitude is in feet

    :return: Partial of pressure in Pa with respect to altitude
    """
    # Implementation for numpy arrays
    idx_tropo = altitude < TROPOPAUSE
    idx_strato = np.logical_not(idx_tropo)

    partial_pressure = np.empty_like(altitude)
    partial_pressure[idx_tropo] = (
        -5.25587611
        / 44330.78
        * SEA_LEVEL_PRESSURE
        * (1 - (altitude[idx_tropo] / 44330.78)) ** 4.25587611
    )
    partial_pressure[idx_strato] = (
        -0.0001576883
        * 22632.0
        * math.log(2.718281)
        * 2.718281 ** (1.7345725 - 0.0001576883 * altitude[idx_strato])
    )

    return partial_pressure * unit_coeff


@compute_partial_pressure.register
@lru_cache()
def _(altitude: Number, unit_coeff) -> float:
    # Implementation for floats
    if altitude < TROPOPAUSE:
        partial_temperature = (
            -5.25587611 / 44330.78 * SEA_LEVEL_PRESSURE * (1 - (altitude / 44330.78)) ** 4.25587611
        )
    else:
        partial_temperature = (
            -0.0001576883
            * 22632.0
            * math.log(2.718281)
            * 2.718281 ** (1.7345725 - 0.0001576883 * altitude)
        )
    return partial_temperature * unit_coeff


# PARTIAL DENSITY =================================================================
def compute_partial_density(
    temperature: Union[np.ndarray, Number],
    pressure: Union[np.ndarray, Number],
    partial_temperature_altitude: Union[np.ndarray, Number],
    partial_pressure_altitude: Union[np.ndarray, Number],
) -> Union[np.ndarray, Number]:
    """
    :param temperature: in K
    :param pressure: in Pa
    :param partial_temperature_altitude: derivative of the temperature in K with respect to the
                                         altitude
    :param partial_pressure_altitude: derivative of the pressure in Pa with respect to the
                                      altitude

    :return: Partial of density in kg/m**3 with respect to altitude
    """

    partial_density = (
        1.0
        / AIR_GAS_CONSTANT
        * (partial_pressure_altitude * temperature - pressure * partial_temperature_altitude)
        / temperature**2.0
    )

    # Unit_coeff is not required because it is already included in the computation of the
    # temperature and pressure partial
    return partial_density


# SPEED OF SOUND =================================================
def compute_partial_speed_of_sound(
    temperature: Union[np.ndarray, Number],
    partial_temperature_altitude: Union[np.ndarray, Number],
) -> Union[np.ndarray, Number]:
    """
    :param temperature: in K
    :param partial_temperature_altitude: derivative of the temperature in K with respect to the
                                         altitude

    :return: Partial of speed of sound in m/s with respect to altitude
    """

    partial_speed_of_sound = (
        0.5 * (1.4 * AIR_GAS_CONSTANT / temperature) ** 0.5 * partial_temperature_altitude
    )

    return partial_speed_of_sound
