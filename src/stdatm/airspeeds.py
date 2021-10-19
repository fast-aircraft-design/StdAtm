"""Conversions between speed parameters."""
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

import numpy as np

from .base import ISpeedConverter, SpeedParameter


class TrueAirspeed(ISpeedConverter):
    """True airspeed in m/s."""

    def compute_value(self, atm):
        """
        Computes true airspeed from available speed data.

        :param atm: the parent Atmosphere instance
        :return: value of true airspeed in m/s
        """
        # Each other descriptor provides the method compute_true_airspeed(), so when the value is
        # needed, we loop on other speed attributes and compute the value from the first
        # not-None value.
        value = None
        for attr, attr_class in SpeedParameter.speed_attributes.items():
            attr_value = getattr(atm, f"_{attr}")
            if attr_value is not None:
                value = attr_class.compute_true_airspeed(atm, attr_value)
                break

        return value

    @staticmethod
    def compute_true_airspeed(atm, value):
        """Needed for inheritance, but unused."""
        return value


class EquivalentAirspeed(ISpeedConverter):
    """Equivalent airspeed in m/s."""

    def compute_value(self, atm):
        """
        Computes equivalent airspeed.

        :param atm: the parent Atmosphere instance
        :return: value of equivalent airspeed in m/s
        """
        if atm.true_airspeed is not None:
            sea_level = type(atm)(0)  # We avoid direct call to Atmosphere to avoid circular import
            return atm.true_airspeed / np.sqrt(sea_level.density / atm.density)

    @staticmethod
    def compute_true_airspeed(atm, value):
        """
        Computes true airspeed from equivalent airspeed.

        :param atm: the parent Atmosphere instance
        :param value: value of equivalent airspeed in m/s
        :return: value of true airspeed in m/s
        """
        sea_level = type(atm)(0)  # We avoid direct call to Atmosphere to avoid circular import
        return value * np.sqrt(sea_level.density / atm.density)


class Mach(ISpeedConverter):
    """Mach number."""

    def compute_value(self, atm):
        """
        Computes Mach number.

        :param atm: the parent Atmosphere instance
        :return: value of Mach number
        """
        if atm.true_airspeed is not None:
            return atm.true_airspeed / atm.speed_of_sound

    @staticmethod
    def compute_true_airspeed(atm, value):
        """
        Computes true airspeed from Mach number.

        :param atm: the parent Atmosphere instance
        :param value: value of Mach number
        :return: value of true airspeed in m/s
        """
        return value * atm.speed_of_sound


class UnitaryReynolds(ISpeedConverter):
    """Unitary Reynolds in 1/m."""

    def compute_value(self, atm):
        """
        Computes unitary Reynolds number.

        :param atm: the parent Atmosphere instance
        :return: value of unitary Reynolds in 1/m
        """
        if atm.true_airspeed is not None:
            return atm.true_airspeed / atm.kinematic_viscosity

    @staticmethod
    def compute_true_airspeed(atm, value):
        """
        Computes true airspeed from unitary Reynolds.

        :param atm: the parent Atmosphere instance
        :param value: value of unitary Reynolds in 1/m
        :return: value of true airspeed in m/s
        """
        return value * atm.kinematic_viscosity


class DynamicPressure(ISpeedConverter):
    """
    Theoretical (true) dynamic pressure in Pa.

    It is given by q = 0.5 * mach**2 * gamma * static_pressure.
    """

    def compute_value(self, atm):
        """
        Computes theoretical (true) dynamic pressure.

        :param atm: the parent Atmosphere instance
        :return: value of dynamic pressure in Pa
        """
        if atm.mach is not None:
            return 0.7 * atm.mach ** 2 * atm.pressure

    @staticmethod
    def compute_true_airspeed(atm, value):
        """
        Computes true airspeed from dynamic pressure.

        :param atm: the parent Atmosphere instance
        :param value: value of dynamic pressure in Pa
        :return: value of true airspeed in m/s
        """
        return np.sqrt(value / 0.7 / atm.pressure) * atm.speed_of_sound
