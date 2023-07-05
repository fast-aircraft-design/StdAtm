"""Functions for computation of atmosphere state parameters."""
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
from numbers import Real
from typing import Union

import numpy as np
from scipy.constants import R, atmosphere

AIR_MOLAR_MASS = 28.9647e-3
AIR_GAS_CONSTANT = R / AIR_MOLAR_MASS
SEA_LEVEL_PRESSURE = atmosphere
SEA_LEVEL_TEMPERATURE = 288.15
TROPOPAUSE = 11000.0
GAMMA = 1.4


# TEMPERATURE =================================================================
@singledispatch
def compute_temperature(altitude, delta_t) -> np.ndarray:
    """

    :param altitude: in m
    :param delta_t: in K
    :return: Temperature in K
    """
    # Implementation for numpy arrays
    idx_tropo = altitude < TROPOPAUSE
    idx_strato = np.logical_not(idx_tropo)

    temperature = np.empty_like(altitude)
    temperature[idx_tropo] = SEA_LEVEL_TEMPERATURE - 0.0065 * altitude[idx_tropo] + delta_t
    temperature[idx_strato] = 216.65 + delta_t

    return temperature


@compute_temperature.register
@lru_cache()
def _(altitude: Real, delta_t: Real) -> float:
    # Implementation for floats
    if altitude < TROPOPAUSE:
        temperature = SEA_LEVEL_TEMPERATURE - 0.0065 * altitude + delta_t
    else:
        temperature = 216.65 + delta_t
    return temperature


# PRESSURE =================================================================
@singledispatch
def compute_pressure(altitude) -> np.ndarray:
    """

    :param altitude: in m
    :return: pressure in Pa
    """
    # Implementation for numpy arrays
    idx_tropo = altitude < TROPOPAUSE
    idx_strato = np.logical_not(idx_tropo)

    pressure = np.empty_like(altitude)
    pressure[idx_tropo] = SEA_LEVEL_PRESSURE * (1 - (altitude[idx_tropo] / 44330.78)) ** 5.25587611
    pressure[idx_strato] = 22632.0 * 2.718281 ** (1.7345725 - 0.0001576883 * altitude[idx_strato])

    return pressure


@compute_pressure.register
@lru_cache()
def _(altitude: Real) -> float:
    # Implementation for floats
    if altitude < TROPOPAUSE:
        pressure = SEA_LEVEL_PRESSURE * (1 - (altitude / 44330.78)) ** 5.25587611
    else:
        pressure = 22632.0 * 2.718281 ** (1.7345725 - 0.0001576883 * altitude)
    return pressure


# DENSITY =================================================================
def compute_density(
    pressure: Union[np.ndarray, Real], temperature: Union[np.ndarray, Real]
) -> Union[np.ndarray, Real]:
    """

    :param pressure: in Pa
    :param temperature: in K
    :return: air density in kg/m**3
    """
    density = pressure / AIR_GAS_CONSTANT / temperature
    return density


# SPEED OF SOUND =================================================
def compute_speed_of_sound(temperature: Union[np.ndarray, Real]) -> Union[np.ndarray, Real]:
    """

    :param temperature: in K
    :return: in m/s
    """
    speed_of_sound = (GAMMA * AIR_GAS_CONSTANT * temperature) ** 0.5
    return speed_of_sound


# DYNAMIC VISCOSITY =================================================
def compute_dynamic_viscosity(temperature: Union[np.ndarray, Real]) -> Union[np.ndarray, Real]:
    """

    :param temperature: in K
    :return: in kg/m/s
    """
    dynamic_viscosity = (0.000017894 * (temperature / SEA_LEVEL_TEMPERATURE) ** 1.5) * (
        (SEA_LEVEL_TEMPERATURE + 110.4) / (temperature + 110.4)
    )

    return dynamic_viscosity


# KINEMATIC VISCOSITY =================================================
def compute_kinematic_viscosity(
    dynamic_viscosity: Union[np.ndarray, Real], density: Union[np.ndarray, Real]
) -> Union[np.ndarray, Real]:
    """

    :param dynamic_viscosity: in kg/m/s
    :param density: in kg/m**3
    :return: in m**2/s
    """
    kinematic_viscosity = dynamic_viscosity / density
    return kinematic_viscosity
