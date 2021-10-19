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

import numpy as np


class ISpeedConverter(ABC):
    """Interface for converting speed values."""

    @abstractmethod
    def compute_value(self, atm):
        """
        Implement here the calculation of the current parameter.

        :param atm: the parent Atmosphere instance
        :return: value of the speed parameter, computed from available data in atm
        """

    @staticmethod
    @abstractmethod
    def compute_true_airspeed(atm, value):
        """
        Implement here the calculation of true airspeed from current parameter.

        :param atm: the parent Atmosphere instance
        :param value: value of the current speed parameter
        :return: value of true airspeed in m/s
        """


class SpeedParameter:
    """
    Descriptor class for speed parameters in
    :class:`~fastoad.model_base.atmosphere.atmosphere.Atmosphere`.
    """

    #: This dict will associate descriptor name (in Atmosphere class) to the descriptor class.
    #: It is useful for doing operations on all speed parameters.
    speed_attributes = {}

    def __init__(self, speed_converter: ISpeedConverter):
        self._converter = speed_converter
        self.__doc__ = self._converter.__doc__  # For correct documentation ins Sphinx

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
        setattr(atm, self.private_name, value)

    def reset_speeds(self, atm):
        """To be used before setting a new speed value as private attribute."""
        for speed_attr in self.speed_attributes:
            setattr(atm, "_" + speed_attr, None)
