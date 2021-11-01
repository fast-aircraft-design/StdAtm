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
from scipy.constants import foot

from .airspeeds import (
    CalibratedAirspeed,
    DynamicPressure,
    EquivalentAirspeed,
    ImpactPressure,
    Mach,
    TrueAirspeed,
    UnitaryReynolds,
)
from .base import SpeedParameter, StaticParameter
from .static_parameters import (
    Density,
    KinematicViscosity,
    Layer,
    Pressure,
    SpeedOfSound,
    Temperature,
)

TROPOPAUSE = 11000


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
