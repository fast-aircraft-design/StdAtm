"""Base classes for atmosphere calculations."""
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

from abc import ABC, abstractmethod
from copy import copy

import numpy as np
from scipy.optimize import fsolve


class AbstractSpeedConverter(ABC):
    """Interface for converting speed values."""

    @abstractmethod
    def compute_value(self, atm):
        """
        Implement here the calculation of the current parameter.

        :param atm: the parent Atmosphere instance
        :return: value of the speed parameter, computed from available data in atm
        """

    def compute_true_airspeed(self, atm, value):
        """
        Computes true airspeed from parameter value.

        This method provides a default implementation that iteratively solves the problem
        using :meth:`compute_value`.

        You may overload this method to provide a direct method.

        :param atm: the parent Atmosphere instance
        :param value: value of the current speed parameter
        :return: value of true airspeed in m/s
        """

        solver_atm = copy(atm)
        shape = np.shape(value)
        value = np.ravel(value)
        root = fsolve(
            lambda tas: value - self._compute_parameter(tas, solver_atm, shape),
            x0=500.0 * np.ones_like(value),
        )
        return np.reshape(root, shape)

    def _compute_parameter(self, tas, atm, shape):
        atm.true_airspeed = np.reshape(tas, shape)
        return np.ravel(self.compute_value(atm))


class SpeedParameter:
    """
    Descriptor class for speed parameters in
    :class:`~fastoad.model_base.atmosphere.atmosphere.Atmosphere`.
    """

    #: This dict will associate descriptor name (in Atmosphere class) to the descriptor class.
    #: It is useful for doing operations on all speed parameters.
    speed_attributes = {}

    def __init__(self, speed_converter: AbstractSpeedConverter):
        self._converter = speed_converter
        self.__doc__ = self._converter.__doc__  # For correct documentation in Sphinx

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name
        self.speed_attributes[name] = type(self._converter)

        setattr(owner, self.private_name, None)

    def __get__(self, atm, owner):
        value = getattr(atm, self.private_name)

        if value is None:
            value = self._converter.compute_value(atm)
            setattr(atm, self.private_name, value)
        return atm.return_value(value)

    def __set__(self, atm, value):
        self.reset_speeds(atm)
        if value is not None:
            value = np.asarray(value)
            try:
                expected_shape = np.shape(value + atm.get_altitude())
            except ValueError as exc:
                raise RuntimeError(
                    f" Shape of provided value for {self.public_name} {value.shape} is not "
                    f"compatible with shape of altitude {atm.get_altitude().shape}."
                ) from exc

            if value.shape != expected_shape:
                value = np.broadcast_to(value, expected_shape)

        setattr(atm, self.private_name, value)

    def reset_speeds(self, atm):
        """To be used before setting a new speed value as private attribute."""
        for speed_attr in self.speed_attributes:
            setattr(atm, "_" + speed_attr, None)
