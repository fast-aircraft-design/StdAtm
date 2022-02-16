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
from copy import deepcopy
from numbers import Number
from typing import Sequence, Union

import numpy as np
from scipy.constants import R, atmosphere, foot
from scipy.optimize import fsolve

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

        # Floats will be provided as output if altitude is a scalar
        self._float_expected = isinstance(altitude, Number)

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
            return self._return_value(self._altitude / foot)
        return self._return_value(self._altitude)

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
        return self._return_value(self._temperature)

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
        return self._return_value(self._pressure)

    @property
    def density(self) -> Union[float, Sequence[float]]:
        """Density in kg/m3."""
        if self._density is None:
            self._density = self.pressure / AIR_GAS_CONSTANT / self.temperature
        return self._return_value(self._density)

    @property
    def speed_of_sound(self) -> Union[float, Sequence[float]]:
        """Speed of sound in m/s."""
        if self._speed_of_sound is None:
            self._speed_of_sound = (1.4 * AIR_GAS_CONSTANT * self.temperature) ** 0.5
        return self._return_value(self._speed_of_sound)

    @property
    def kinematic_viscosity(self) -> Union[float, Sequence[float]]:
        """Kinematic viscosity in m2/s."""
        if self._kinematic_viscosity is None:
            self._kinematic_viscosity = (
                (0.000017894 * (self.temperature / SEA_LEVEL_TEMPERATURE) ** (3 / 2))
                * ((SEA_LEVEL_TEMPERATURE + 110.4) / (self.temperature + 110.4))
            ) / self.density
        return self._return_value(self._kinematic_viscosity)

    @property
    def mach(self) -> Union[float, Sequence[float]]:
        """Mach number."""
        if self._mach is None and self.true_airspeed is not None:
            self._mach = self.true_airspeed / self.speed_of_sound
        return self._return_value(self._mach)

    @property
    def true_airspeed(self) -> Union[float, Sequence[float]]:
        """True airspeed (TAS) in m/s."""
        # Dev note: true_airspeed is the "hub". Other speed values will be calculated
        # from this true_airspeed.
        if self._true_airspeed is None:
            if self._mach is not None:
                self._true_airspeed = self._mach * self.speed_of_sound
            elif self._equivalent_airspeed is not None:
                sea_level = Atmosphere(0)
                self._true_airspeed = self._equivalent_airspeed * np.sqrt(
                    sea_level.density / self.density
                )
            elif self._unitary_reynolds is not None:
                self._true_airspeed = self._unitary_reynolds * self.kinematic_viscosity
            elif self._dynamic_pressure is not None:
                self._true_airspeed = (
                    np.sqrt(self._dynamic_pressure / 0.7 / self.pressure) * self.speed_of_sound
                )
            elif self._impact_pressure is not None:
                self._true_airspeed = self._compute_true_airspeed(
                    "impact_pressure", self._impact_pressure
                )
            elif self._calibrated_airspeed is not None:
                self._true_airspeed = self._compute_true_airspeed(
                    "calibrated_airspeed", self._calibrated_airspeed
                )

        return self._return_value(self._true_airspeed)

    @property
    def equivalent_airspeed(self) -> Union[float, Sequence[float]]:
        """Equivalent airspeed (EAS) in m/s."""
        if self._equivalent_airspeed is None and self.true_airspeed is not None:
            sea_level = Atmosphere(0)
            self._equivalent_airspeed = self.true_airspeed / np.sqrt(
                sea_level.density / self.density
            )

        return self._return_value(self._equivalent_airspeed)

    @property
    def unitary_reynolds(self) -> Union[float, Sequence[float]]:
        """Unitary Reynolds number in 1/m."""
        if self._unitary_reynolds is None and self.true_airspeed is not None:
            self._unitary_reynolds = self.true_airspeed / self.kinematic_viscosity
        return self._return_value(self._unitary_reynolds)

    @property
    def dynamic_pressure(self) -> Union[float, Sequence[float]]:
        """
        Theoretical (true) dynamic pressure in Pa.

        It is given by q = 0.5 * mach**2 * gamma * static_pressure.
        """

        if self.mach is not None:
            self._dynamic_pressure = 0.7 * self.mach**2 * self.pressure
        return self._return_value(self._dynamic_pressure)

    @property
    def impact_pressure(self) -> Union[float, Sequence[float]]:
        """Compressible dynamic pressure in Pa."""

        def _compute_subsonic_impact_pressure(mach, p):
            return p * ((1 + 0.2 * mach**2) ** 3.5 - 1)

        def _compute_supersonic_impact_pressure(mach, p):
            # Rayleigh law
            # https://en.wikipedia.org/wiki/Rayleigh_flow#Additional_Rayleigh_Flow_Relations
            return p * (166.92158 * mach**7 / (7 * mach**2 - 1) ** 2.5 - 1)

        if self.mach is not None:
            mach = np.asarray(self.mach)
            idx_subsonic = mach <= 1.0
            idx_supersonic = mach > 1

            if np.shape(self.pressure) != np.shape(mach):
                pressure = np.broadcast_to(self.pressure, np.shape(mach))
            else:
                pressure = np.asarray(self.pressure)

            value = np.empty_like(mach)
            value[idx_subsonic] = _compute_subsonic_impact_pressure(
                mach[idx_subsonic], pressure[idx_subsonic]
            )
            value[idx_supersonic] = _compute_supersonic_impact_pressure(
                mach[idx_supersonic], pressure[idx_supersonic]
            )
            self._impact_pressure = value
            return self._return_value(self._impact_pressure)

    @property
    def calibrated_airspeed(self):
        """Calibrated airspeed in m/s."""
        #         Computation is done using Eq. 3.16 and 3.17 from:
        #         Gracey, William (1980), "Measurement of Aircraft Speed and Altitude",
        #         NASA Reference Publication 1046.
        #         https://apps.dtic.mil/sti/pdfs/ADA280006.pdf

        def _compute_cas_low_speed(impact_pressure, sea_level):
            return sea_level.speed_of_sound * np.sqrt(
                5 * ((impact_pressure / sea_level.pressure + 1) ** (1 / 3.5) - 1)
            )

        def _compute_cas_high_speed(impact_pressure, sea_level):
            root = fsolve(
                _equation_cas_high_speed,
                x0=sea_level.speed_of_sound * np.ones_like(impact_pressure),
                args=(impact_pressure, sea_level),
            )
            return root

        def _equation_cas_high_speed(cas, impact_pressure, sea_level):
            return cas - sea_level.speed_of_sound * (
                (impact_pressure / sea_level.pressure + 1)
                * (7 * (cas / sea_level.speed_of_sound) ** 2 - 1) ** 2.5
                / (6**2.5 * 1.2**3.5)
            ) ** (1 / 7)

        if self.impact_pressure is not None:
            sea_level = Atmosphere(0)
            impact_pressure = np.asarray(self.impact_pressure)

            cas = np.asarray(_compute_cas_low_speed(impact_pressure, sea_level))
            idx_high_speed = cas > sea_level.speed_of_sound
            if np.any(idx_high_speed):
                cas[idx_high_speed] = _compute_cas_high_speed(
                    impact_pressure[idx_high_speed], sea_level
                )

            self._calibrated_airspeed = cas
            return self._return_value(self._calibrated_airspeed)

    @mach.setter
    def mach(self, value: Union[float, Sequence[float]]):
        self._reset_speeds()
        if value is not None:
            self._mach = self._adapt_shape(value)

    @true_airspeed.setter
    def true_airspeed(self, value: Union[float, Sequence[float]]):
        self._reset_speeds()
        if value is not None:
            self._true_airspeed = self._adapt_shape(value)

    @equivalent_airspeed.setter
    def equivalent_airspeed(self, value: Union[float, Sequence[float]]):
        self._reset_speeds()
        if value is not None:
            self._equivalent_airspeed = self._adapt_shape(value)

    @unitary_reynolds.setter
    def unitary_reynolds(self, value: Union[float, Sequence[float]]):
        self._reset_speeds()
        if value is not None:
            self._unitary_reynolds = self._adapt_shape(value)

    @dynamic_pressure.setter
    def dynamic_pressure(self, value: Union[float, Sequence[float]]):
        self._reset_speeds()
        if value is not None:
            self._dynamic_pressure = self._adapt_shape(value)

    @impact_pressure.setter
    def impact_pressure(self, value: Union[float, Sequence[float]]):
        self._reset_speeds()
        if value is not None:
            self._impact_pressure = self._adapt_shape(value)

    @calibrated_airspeed.setter
    def calibrated_airspeed(self, value: Union[float, Sequence[float]]):
        self._reset_speeds()
        if value is not None:
            self._calibrated_airspeed = self._adapt_shape(value)

    def _adapt_shape(self, value):
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

    def _return_value(self, value):
        """
        :returns: a float when needed. Otherwise, returns the value itself.
        """
        if self._float_expected and value is not None:
            try:
                # It's faster to try... catch than to test np.size(value).
                # (but float(value) is slow to fail if value is None, so
                #  it is why we test it before)
                return float(value)
            except TypeError:
                pass
        return value

    def _compute_true_airspeed(self, parameter_name, value):
        """
        Computes true airspeed from parameter value.

        This method provides a default implementation that iteratively solves the problem
        using :meth:`compute_value`.

        You may overload this method to provide a direct method.

        :param atm: the parent Atmosphere instance
        :param value: value of the current speed parameter
        :return: value of true airspeed in m/s
        """

        def _compute_parameter(tas, atm, shape):
            atm.true_airspeed = np.reshape(tas, shape)
            return np.ravel(getattr(atm, parameter_name))

        solver_atm = deepcopy(self)
        shape = np.shape(value)
        value = np.ravel(value)
        root = fsolve(
            lambda tas: value - _compute_parameter(tas, solver_atm, shape),
            x0=500.0 * np.ones_like(value),
        )
        return np.reshape(root, shape)


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
