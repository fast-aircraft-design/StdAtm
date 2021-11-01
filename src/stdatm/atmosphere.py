"""
Simple implementation of International Standard Atmosphere.
"""
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

from numbers import Number
from typing import Sequence, Union

import numpy as np
from scipy.constants import R, atmosphere, foot, g

from .airspeeds import (
    CalibratedAirspeed,
    DynamicPressure,
    EquivalentAirspeed,
    ImpactPressure,
    Mach,
    TrueAirspeed,
    UnitaryReynolds,
)
from .base import AbstractStaticCalculator, SpeedParameter, StaticParameter

TROPOPAUSE = 11000


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
        return np.maximum(0, np.searchsorted(LAYERS["altitude"], atm.altitude) - 1)


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
        pressure = np.zeros_like(atm.altitude)

        idx = atm.altitude == 0.0
        pressure[idx] = SEA_LEVEL_PRESSURE

        idx = np.logical_not(idx)
        if np.any(idx):
            base_altitude = LAYERS["altitude"][atm.layer]
            base = AtmosphereSI(base_altitude, delta_t=atm.delta_t)

            idx_gradient_0 = np.logical_and(idx, LAYERS["temperature_gradient"][atm.layer] == 0.0)

            if np.any(idx_gradient_0):
                p_b = np.asarray(base.pressure)[idx_gradient_0]
                t = np.asarray(atm.temperature)[idx_gradient_0]
                h = np.asarray(atm.altitude)[idx_gradient_0]
                h_b = np.asarray(base.altitude)[idx_gradient_0]
                pressure[idx_gradient_0] = p_b * np.exp(-g / AIR_GAS_CONSTANT / t * (h - h_b))

            idx_gradient_non0 = np.logical_not(idx_gradient_0)
            if np.any(idx_gradient_non0):
                p_b = np.asarray(base.pressure)[idx_gradient_non0]
                beta = LAYERS["temperature_gradient"][atm.layer][idx_gradient_non0]
                t_b = np.asarray(base.temperature)[idx_gradient_non0]
                h = np.asarray(atm.altitude)[idx_gradient_non0]
                h_b = np.asarray(base.altitude)[idx_gradient_non0]
                pressure[idx_gradient_non0] = p_b * (1.0 + beta / t_b * (h - h_b)) ** (
                    -g / AIR_GAS_CONSTANT / beta
                )

        # idx_tropo = atm.layer == 0
        # idx_strato = atm.layer == 1
        # pressure[idx_tropo] = (
        #     SEA_LEVEL_PRESSURE * (1 - (atm._altitude[idx_tropo] / 44330.78)) ** 5.25587611
        # )
        # pressure[idx_strato] = 22632 * 2.718281 ** (
        #     1.7345725 - 0.0001576883 * atm._altitude[idx_strato]
        # )
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


class AtmosphereSI:
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
        >>> pressure = AtmosphereSI(10000).pressure # pressure at 10,000 m, dISA = 0 K
        >>> density = AtmosphereSI(5000, 10).density # density at 5,000 m, dISA = 10 K


        >>> atm = AtmosphereSI([0.0,10000.0,30000.0]) # init for alt. 0, 10,000 and 30,000 feet
        >>> atm.pressure # pressures for all defined altitudes
        array([101325.        ,  69681.66657158,  30089.59825871])
        >>> atm.kinematic_viscosity # viscosities for all defined altitudes
        array([1.46074563e-05, 1.87057660e-05, 3.24486943e-05])

    Also, after instantiating this class, setting one speed parameter allows to get value of other
    ones.
    Provided speed values should have a shape compatible with provided altitudes.

    .. code-block::

        >>> atm1 = AtmosphereSI(10000)
        >>> atm1.true_airspeed = [100.0, 250.0]
        >>> atm1.mach
        array([0.33404192, 0.83510481])

        >>> atm2 = AtmosphereSI([0, 1000, 11000])
        >>> atm2.equivalent_airspeed = 200.0
        >>> atm2.true_airspeed
        array([200.        , 209.76266034, 366.48606529])

        >>> atm2.mach = [1.0, 1.5, 2.0]
        >>> atm2.true_airspeed
        array([340.20668009, 504.07018698, 589.255255  ])

        >>> atm2.equivalent_airspeed = [[300, 200, 100],[50, 100, 150]]
        >>> atm2.true_airspeed
        array([[300.        , 209.76266034, 183.24303265],
               [ 50.        , 104.88133017, 274.86454897]])
    """

    # Descriptors for static parameters
    layer = StaticParameter(Layer())
    temperature = StaticParameter(Temperature())
    pressure = StaticParameter(Pressure())
    density = StaticParameter(Density())
    speed_of_sound = StaticParameter(SpeedOfSound())
    kinematic_viscosity = StaticParameter(KinematicViscosity())

    # Descriptors for speed parameters
    true_airspeed = SpeedParameter(TrueAirspeed())
    equivalent_airspeed = SpeedParameter(EquivalentAirspeed())
    calibrated_airspeed = SpeedParameter(CalibratedAirspeed())
    mach = SpeedParameter(Mach())
    unitary_reynolds = SpeedParameter(UnitaryReynolds())
    dynamic_pressure = SpeedParameter(DynamicPressure())
    impact_pressure = SpeedParameter(ImpactPressure())

    # pylint: disable=too-many-instance-attributes  # Needed for avoiding redoing computations
    def __init__(
        self,
        altitude: Union[Number, Sequence],
        delta_t: Number = 0.0,
    ):
        """
        :param altitude: altitude in meters.
        :param delta_t: temperature increment (K) applied to whole temperature profile
        """

        self.altitude = altitude
        self.delta_t = delta_t

    @property
    def altitude(self):
        """Altitude in meters."""
        return self.return_value(self._altitude)

    @altitude.setter
    def altitude(self, value: Union[float, Sequence[float]]):
        # Floats will be provided as output if altitude is a scalar
        self._scalar_expected = isinstance(value, Number)

        self._altitude = np.asarray(value, dtype=np.float64)

    def get_altitude(self, altitude_in_feet: bool = True) -> Union[float, Sequence[float]]:
        """
        Convenience method that provides by default altitude in feet.

        :code:`.get_altitude(False)` is equivalent to :code:`.altitude`.

        :param altitude_in_feet: if True, altitude is returned in feet. Otherwise,
                                 it is returned in meters
        :return: altitude in feet or in meters
        """
        if altitude_in_feet:
            return self.altitude / foot
        return self.altitude

    @property
    def delta_t(self) -> Union[float, Sequence[float]]:
        """Temperature increment applied to whole temperature profile."""
        return self.return_value(self._delta_t)

    @delta_t.setter
    def delta_t(self, value: Union[float, Sequence[float]]):
        self._delta_t = np.asarray(value, dtype=np.float64)

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


class Atmosphere(AtmosphereSI):
    """
    Same as :class:`AtmosphereSI` except that allows to instantiate with altitude in feet.

    .. Warning::

        Property :attr:`AtmosphereSI.altitude` will still provide altitude in meters.
        Use :meth:`~AtmosphereSI.get_altitude` to get altitude in feet.
    """

    def __init__(
        self,
        altitude: Union[float, Sequence[float]],
        delta_t: float = 0.0,
        altitude_in_feet: bool = True,
    ):
        """
        :param altitude: altitude in meters
        :param delta_t: temperature increment (°C) applied to whole temperature profile
        :param altitude_in_feet: if True, altitude should be provided in feet. Otherwise,
                                 it should be provided in meters.

        """
        super().__init__(altitude, delta_t)

        if altitude_in_feet:
            self.altitude *= foot
