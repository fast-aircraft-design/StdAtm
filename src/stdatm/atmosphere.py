"""
Simple implementation of International Standard Atmosphere.
"""
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

from copy import deepcopy
from numbers import Number
from typing import Sequence, Union

import numpy as np
from scipy.constants import foot
from scipy.optimize import root

from stdatm.speed_parameters import (
    compute_calibrated_airspeed,
    compute_dynamic_pressure,
    compute_equivalent_airspeed,
    compute_impact_pressure,
    compute_mach,
    compute_tas_from_eas,
    compute_tas_from_mach,
    compute_tas_from_pdyn,
    compute_tas_from_unit_re,
    compute_unitary_reynolds,
)
from stdatm.state_parameters import (
    compute_density,
    compute_dynamic_viscosity,
    compute_kinematic_viscosity,
    compute_pressure,
    compute_speed_of_sound,
    compute_temperature,
)


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

    # pylint: disable=too-many-instance-attributes  # Needed for avoiding redoing computations
    def __init__(
        self,
        altitude: Union[float, Sequence[float]],
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

        # For convenience, let's have altitude as numpy arrays and in meters in all cases
        unit_coeff = foot if altitude_in_feet else 1.0
        if not isinstance(altitude, Number):
            altitude = np.asarray(altitude)
        self._altitude = altitude * unit_coeff

        # Outputs
        self._temperature = None
        self._pressure = None
        self._density = None
        self._speed_of_sound = None
        self._dynamic_viscosity = None
        self._kinematic_viscosity = None
        self._mach = None
        self._equivalent_airspeed = None
        self._true_airspeed = None
        self._unitary_reynolds = None
        self._dynamic_pressure = None
        self._impact_pressure = None
        self._calibrated_airspeed = None

    def get_altitude(self, altitude_in_feet: bool = True) -> Union[float, Sequence[float]]:
        """
        :param altitude_in_feet: if True, altitude is returned in feet. Otherwise,
                                 it is returned in meters
        :return: altitude provided at instantiation
        """
        if altitude_in_feet:
            return self._altitude / foot
        return self._altitude

    @property
    def delta_t(self) -> float:
        """Temperature increment applied to whole temperature profile."""
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value: float):
        # Let's ensure it is not a one-element array that would crash lru_cache
        if isinstance(value, Number):
            self._delta_t = value
        else:
            self._delta_t = np.asarray(value).item()

    @property
    def temperature(self) -> Union[float, np.ndarray]:
        """Temperature in K."""
        if self._temperature is None:
            self._temperature = compute_temperature(self._altitude, self.delta_t)

        return self._temperature

    @property
    def pressure(self) -> Union[float, np.ndarray]:
        """Pressure in Pa."""
        if self._pressure is None:
            self._pressure = compute_pressure(self._altitude)

        return self._pressure

    @property
    def density(self) -> Union[float, np.ndarray]:
        """Density in kg/m3."""
        if self._density is None:
            self._density = compute_density(self.pressure, self.temperature)
        return self._density

    @property
    def speed_of_sound(self) -> Union[float, np.ndarray]:
        """Speed of sound in m/s."""
        if self._speed_of_sound is None:
            self._speed_of_sound = compute_speed_of_sound(self.temperature)
        return self._speed_of_sound

    @property
    def dynamic_viscosity(self) -> Union[float, np.ndarray]:
        """Dynamic viscosity in kg/m/s."""
        if self._dynamic_viscosity is None:
            self._dynamic_viscosity = compute_dynamic_viscosity(self.temperature)
        return self._dynamic_viscosity

    @property
    def kinematic_viscosity(self) -> Union[float, np.ndarray]:
        """Kinematic viscosity in m2/s."""
        if self._kinematic_viscosity is None:
            self._kinematic_viscosity = compute_kinematic_viscosity(
                self.dynamic_viscosity, self.density
            )
        return self._kinematic_viscosity

    @property
    def true_airspeed(self) -> Union[float, np.ndarray]:
        """True airspeed (TAS) in m/s."""
        # Dev note: true_airspeed is the "hub". Other speed values will be calculated
        # from this true_airspeed.
        if self._true_airspeed is None:
            if self._mach is not None:
                self._true_airspeed = compute_tas_from_mach(self._mach, self.speed_of_sound)
            elif self._equivalent_airspeed is not None:
                self._true_airspeed = compute_tas_from_eas(self._equivalent_airspeed, self.density)
            elif self._unitary_reynolds is not None:
                self._true_airspeed = compute_tas_from_unit_re(
                    self._unitary_reynolds, self.kinematic_viscosity
                )
            elif self._dynamic_pressure is not None:
                self._true_airspeed = compute_tas_from_pdyn(self._dynamic_pressure, self.density)
            elif self._impact_pressure is not None:
                self._true_airspeed = self._compute_true_airspeed(
                    "impact_pressure", self._impact_pressure
                )
            elif self._calibrated_airspeed is not None:
                self._true_airspeed = self._compute_true_airspeed(
                    "calibrated_airspeed", self._calibrated_airspeed
                )

        return self._true_airspeed

    @property
    def mach(self) -> Union[float, np.ndarray]:
        """Mach number."""
        if self._mach is None and self.true_airspeed is not None:
            self._mach = compute_mach(self.true_airspeed, self.speed_of_sound)
        return self._mach

    @property
    def equivalent_airspeed(self) -> Union[float, np.ndarray]:
        """Equivalent airspeed (EAS) in m/s."""
        if self._equivalent_airspeed is None and self.true_airspeed is not None:
            self._equivalent_airspeed = compute_equivalent_airspeed(
                self.true_airspeed, self.density
            )
        return self._equivalent_airspeed

    @property
    def unitary_reynolds(self) -> Union[float, np.ndarray]:
        """Unitary Reynolds number in 1/m."""
        if self._unitary_reynolds is None and self.true_airspeed is not None:
            self._unitary_reynolds = compute_unitary_reynolds(
                self.true_airspeed, self.kinematic_viscosity
            )
        return self._unitary_reynolds

    @property
    def dynamic_pressure(self) -> Union[float, np.ndarray]:
        """
        Theoretical (true) dynamic pressure in Pa.

        It is given by q = 0.5 * mach**2 * gamma * static_pressure.
        """
        if self.mach is not None:
            self._dynamic_pressure = compute_dynamic_pressure(self.true_airspeed, self.density)
        return self._dynamic_pressure

    @property
    def impact_pressure(self) -> Union[float, np.ndarray]:
        """Compressible dynamic pressure (AKA impact pressure) in Pa."""
        if self.mach is not None:
            self._impact_pressure = compute_impact_pressure(self.mach, self.pressure)
        return self._impact_pressure

    @property
    def calibrated_airspeed(self) -> Union[float, np.ndarray]:
        """Calibrated airspeed in m/s."""
        if self.impact_pressure is not None:
            self._calibrated_airspeed = compute_calibrated_airspeed(
                self.impact_pressure,
            )
        return self._calibrated_airspeed

    @mach.setter
    def mach(self, value: Union[float, Sequence[float]]):
        self._mach = self._process_speed_value(value)

    @true_airspeed.setter
    def true_airspeed(self, value: Union[float, Sequence[float]]):
        self._true_airspeed = self._process_speed_value(value)

    @equivalent_airspeed.setter
    def equivalent_airspeed(self, value: Union[float, Sequence[float]]):
        self._equivalent_airspeed = self._process_speed_value(value)

    @unitary_reynolds.setter
    def unitary_reynolds(self, value: Union[float, Sequence[float]]):
        self._unitary_reynolds = self._process_speed_value(value)

    @dynamic_pressure.setter
    def dynamic_pressure(self, value: Union[float, Sequence[float]]):
        self._dynamic_pressure = self._process_speed_value(value)

    @impact_pressure.setter
    def impact_pressure(self, value: Union[float, Sequence[float]]):
        self._impact_pressure = self._process_speed_value(value)

    @calibrated_airspeed.setter
    def calibrated_airspeed(self, value: Union[float, Sequence[float]]):
        self._calibrated_airspeed = self._process_speed_value(value)

    def _process_speed_value(self, value):
        self._reset_speeds()
        return self._adapt_shape(value)

    def _adapt_shape(self, value):
        if value is not None:
            value = np.asarray(value)
            if np.size(value) > 1:
                try:
                    expected_shape = np.shape(value + self.get_altitude())
                except ValueError as exc:
                    raise RuntimeError(
                        "Shape of provided value is not "
                        f"compatible with shape of altitude {np.shape(self.get_altitude())}."
                    ) from exc

                if value.shape != expected_shape:
                    value = np.broadcast_to(value, expected_shape)

        return value

    def _reset_speeds(self):
        """To be used before setting a new speed value as private attribute."""
        self._mach = None
        self._true_airspeed = None
        self._equivalent_airspeed = None
        self._unitary_reynolds = None
        self._dynamic_pressure = None
        self._impact_pressure = None
        self._calibrated_airspeed = None

    def _compute_true_airspeed(self, parameter_name, value):
        """
        Computes true airspeed from parameter value.

        This method provides a default implementation that iteratively solves the problem
        using :meth:`compute_value`.

        You may overload this method to provide a direct method.

        :param parameter_name: name of the input speed attribute
        :param value: value of the input speed parameter
        :return: value of true airspeed in m/s
        """

        def _compute_parameter(tas, atm, shape):
            atm.true_airspeed = np.reshape(tas, shape)
            return np.ravel(getattr(atm, parameter_name))

        solver_atm = deepcopy(self)
        shape = np.shape(value)
        value = np.ravel(value)
        solution = root(
            lambda tas: value - _compute_parameter(tas, solver_atm, shape),
            x0=500.0 * np.ones_like(value),
        )
        return np.reshape(solution.x, shape)


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
