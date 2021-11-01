#  This file is part of StdAtm
#  Copyright (C) 2022 ONERA & ISAE-SUPAERO
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

import numpy as np
from scipy.constants import R, atmosphere

from .base import AbstractStaticCalculator

AIR_MOLAR_MASS = 28.9647e-3
AIR_GAS_CONSTANT = R / AIR_MOLAR_MASS
SEA_LEVEL_PRESSURE = atmosphere
SEA_LEVEL_TEMPERATURE = 288.15
LAYERS = np.array(
    [
        (0.0, 288.15, -6.5e-3),
        (11000.0, 216.65, 0.0),
        (1e6, np.nan, np.nan),
    ],
    dtype=[("altitude", "f8"), ("base_temperature", "f8"), ("temperature_gradient", "f8")],
)


class Layer(AbstractStaticCalculator):
    def compute_value(self, atm):
        return np.maximum(0, np.searchsorted(LAYERS["altitude"], atm.altitude, side="right") - 1)


class Temperature(AbstractStaticCalculator):
    """Air temperature in K."""

    def compute_value(self, atm):
        """
        Computes air temperature.

        :param atm: the parent Atmosphere instance
        :return: value of air temperature in K
        """
        temperature = (
            LAYERS["base_temperature"][atm.layer]
            + LAYERS["temperature_gradient"][atm.layer]
            * (atm.altitude - LAYERS["altitude"][atm.layer])
            + atm.delta_t
        )
        return temperature


class Pressure(AbstractStaticCalculator):
    """Pressure in Pa."""

    def compute_value(self, atm):
        """
        Computes air pressure.

        :param atm: the parent Atmosphere instance
        :return: value of air pressure in Pa
        """
        pressure = np.zeros_like(atm._altitude)
        idx_tropo = atm.layer == 0
        idx_strato = atm.layer == 1
        pressure[idx_tropo] = (
            SEA_LEVEL_PRESSURE * (1 - (atm._altitude[idx_tropo] / 44330.78)) ** 5.25587611
        )
        pressure[idx_strato] = 22632 * 2.718281 ** (
            1.7345725 - 0.0001576883 * atm._altitude[idx_strato]
        )
        return pressure


class Density(AbstractStaticCalculator):
    """Air density in kg/m**3."""

    def compute_value(self, atm):
        """
        Computes air density.

        :param atm: the parent Atmosphere instance
        :return: value of air density in kg/m**3
        """
        return atm.pressure / AIR_GAS_CONSTANT / atm.temperature


class SpeedOfSound(AbstractStaticCalculator):
    """Speed of sound in m/s."""

    def compute_value(self, atm):
        """
        Computes speed of sound.

        :param atm: the parent Atmosphere instance
        :return: value of speed of sound in m/s
        """
        return (1.4 * AIR_GAS_CONSTANT * atm.temperature) ** 0.5


class KinematicViscosity(AbstractStaticCalculator):
    """Kinematic viscosity in m**2/s."""

    def compute_value(self, atm):
        """
        Computes kinematic viscosity.

        :param atm: the parent Atmosphere instance
        :return: value of kinematic viscosity in m**2/s
        """
        return (
            (0.000017894 * (atm.temperature / SEA_LEVEL_TEMPERATURE) ** (3 / 2))
            * ((SEA_LEVEL_TEMPERATURE + 110.4) / (atm.temperature + 110.4))
        ) / atm.density
