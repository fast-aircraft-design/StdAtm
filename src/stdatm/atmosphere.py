"""
Simple implementation of International Standard Atmosphere.
"""
#  This file is part of StdAtm
#  Copyright (C) 2021 ONERA & ISAE-SUPAERO
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

from numbers import Number
from typing import Sequence, Union

import numpy as np
from scipy.constants import R, atmosphere, foot

from .airspeeds import EquivalentAirspeed, Mach, TrueAirspeed, UnitaryReynolds

AIR_MOLAR_MASS = 28.9647e-3
AIR_GAS_CONSTANT = R / AIR_MOLAR_MASS
SEA_LEVEL_PRESSURE = atmosphere
SEA_LEVEL_TEMPERATURE = 288.15
TROPOPAUSE = 11000


class Atmosphere:
    """
    Simple implementation of International Standard Atmosphere
    for troposphere and stratosphere.

    Atmosphere properties are provided in the same "shape" as provided
    altitude:

    - if altitude is given as a float, returned values will be floats
    - if altitude is given as a sequence (list, 1D numpy array, ...), returned
      values will be 1D numpy arrays
    - if altitude is given as nD numpy array, returned values will be nD numpy
      arrays

    Usage:

    .. code-block::
        >>> from stdatm import Atmosphere
        >>> pressure = Atmosphere(30000).pressure # pressure at 30,000 feet, dISA = 0 K
        >>> density = Atmosphere(5000, 10).density # density at 5,000 feet, dISA = 10 K


        >>> atm = Atmosphere([0.0,10000.0,30000.0]) # init for alt. 0, 10,000 and 30,000 feet
        >>> atm.pressure # pressures for all defined altitudes
        array([101325.        ,  69681.66657158,  30089.59825871])
        >>> atm.kinematic_viscosity # viscosities for all defined altitudes
        array([1.46074563e-05, 1.87057660e-05, 3.24486943e-05])

    Also, after instantiating this class, setting one speed parameter allows to get value of other
    ones.
    Provided speed values should have a shape compatible with provided altitudes.

    .. code-block::

        >>> atm1 = Atmosphere(30000)
        >>> atm1.true_airspeed = [100.0, 250.0]
        >>> atm1.mach
        array([0.32984282, 0.82460705])

        >>> atm2 = Atmosphere([0, 1000, 35000])
        >>> atm2.equivalent_airspeed = 200.0
        >>> atm2.true_airspeed
        array([200.        , 202.95792913, 359.28282052])

        >>> atm2.mach = [1.0, 1.5, 2.0]
        >>> atm2.true_airspeed
        array([340.29526405, 508.68507243, 593.0730464 ])

        >>> atm2.equivalent_airspeed = [[300, 200, 100],[50, 100, 150]]
        >>> atm2.true_airspeed
        array([[300.        , 202.95792913, 179.64141026],
               [ 50.        , 101.47896457, 269.46211539]])
    """

    # Descriptors for speed conversions
    true_airspeed = TrueAirspeed()
    equivalent_airspeed = EquivalentAirspeed()
    mach = Mach()
    unitary_reynolds = UnitaryReynolds()

    # pylint: disable=too-many-instance-attributes  # Needed for avoiding redoing computations
    def __init__(
        self,
        altitude: Union[float, Sequence],
        delta_t: float = 0.0,
        altitude_in_feet: bool = True,
    ):
        """
        :param altitude: altitude (units decided by altitude_in_feet)
        :param delta_t: temperature increment (°C) applied to whole temperature profile
        :param altitude_in_feet: if True, altitude should be provided in feet. Otherwise,
                                 it should be provided in meters.
        """

        self.delta_t = delta_t

        # Floats will be provided as output if altitude is a scalar
        self._scalar_expected = isinstance(altitude, Number)

        # For convenience, let's have altitude as numpy arrays and in meters in all cases
        unit_coeff = foot if altitude_in_feet else 1.0
        self._altitude = np.asarray(altitude) * unit_coeff

        # Sets indices for tropopause
        self._idx_tropo = self._altitude < TROPOPAUSE
        self._idx_strato = self._altitude >= TROPOPAUSE

        # Outputs
        self._temperature = None
        self._pressure = None
        self._density = None
        self._speed_of_sound = None
        self._kinematic_viscosity = None

    def get_altitude(self, altitude_in_feet: bool = True) -> Union[float, Sequence[float]]:
        """
        :param altitude_in_feet: if True, altitude is returned in feet. Otherwise,
                                 it is returned in meters
        :return: altitude provided at instantiation
        """
        if altitude_in_feet:
            return self.return_value(self._altitude / foot)
        return self.return_value(self._altitude)

    @property
    def delta_t(self) -> Union[float, Sequence[float]]:
        """Temperature increment applied to whole temperature profile."""
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value: Union[float, Sequence[float]]):
        self._delta_t = np.asarray(value)

    @property
    def temperature(self) -> Union[float, Sequence[float]]:
        """Temperature in K."""
        if self._temperature is None:
            self._temperature = np.zeros(self._altitude.shape)
            self._temperature[self._idx_tropo] = (
                SEA_LEVEL_TEMPERATURE - 0.0065 * self._altitude[self._idx_tropo] + self._delta_t
            )
            self._temperature[self._idx_strato] = 216.65 + self._delta_t
        return self.return_value(self._temperature)

    @property
    def pressure(self) -> Union[float, Sequence[float]]:
        """Pressure in Pa."""
        if self._pressure is None:
            self._pressure = np.zeros(self._altitude.shape)
            self._pressure[self._idx_tropo] = (
                SEA_LEVEL_PRESSURE
                * (1 - (self._altitude[self._idx_tropo] / 44330.78)) ** 5.25587611
            )
            self._pressure[self._idx_strato] = 22632 * 2.718281 ** (
                1.7345725 - 0.0001576883 * self._altitude[self._idx_strato]
            )
        return self.return_value(self._pressure)

    @property
    def density(self) -> Union[float, Sequence[float]]:
        """Density in kg/m3."""
        if self._density is None:
            self._density = self.pressure / AIR_GAS_CONSTANT / self.temperature
        return self.return_value(self._density)

    @property
    def speed_of_sound(self) -> Union[float, Sequence[float]]:
        """Speed of sound in m/s."""
        if self._speed_of_sound is None:
            self._speed_of_sound = (1.4 * AIR_GAS_CONSTANT * self.temperature) ** 0.5
        return self.return_value(self._speed_of_sound)

    @property
    def kinematic_viscosity(self) -> Union[float, Sequence[float]]:
        """Kinematic viscosity in m2/s."""
        if self._kinematic_viscosity is None:
            self._kinematic_viscosity = (
                (0.000017894 * (self.temperature / SEA_LEVEL_TEMPERATURE) ** (3 / 2))
                * ((SEA_LEVEL_TEMPERATURE + 110.4) / (self.temperature + 110.4))
            ) / self.density
        return self.return_value(self._kinematic_viscosity)

    def return_value(self, value):
        """
        :returns: a scalar when needed. Otherwise, returns the value itself.
        """
        if self._scalar_expected and value is not None:
            try:
                # It's faster to try... catch than to test np.size(value).
                # (but float(value) is slow to fail if value is None, so
                # it is why we test it before)
                return np.asarray(value).item()
            except ValueError:
                pass
        return value


class AtmosphereSI(Atmosphere):
    """Same as :class:`Atmosphere` except that altitudes are always in meters."""

    def __init__(self, altitude: Union[float, Sequence[float]], delta_t: float = 0.0):
        """
        :param altitude: altitude in meters
        :param delta_t: temperature increment (°C) applied to whole temperature profile
        """
        super().__init__(altitude, delta_t, altitude_in_feet=False)

    @property
    def altitude(self):
        """Altitude in meters."""
        return self.get_altitude(altitude_in_feet=False)
