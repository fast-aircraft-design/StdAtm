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


class SpeedConverter(ABC):
    """
    Base descriptor class for speed parameters in
    :class:`~fastoad.model_base.atmosphere.atmosphere.Atmosphere`.
    """

    # This dict will associate descriptor name (in Atmosphere class) to the descriptor class.
    # It is useful for doing operations on all speed parameters.
    _speed_attributes = {}

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name
        self._speed_attributes[name] = self.__class__

    def __get__(self, atm, owner):
        value = self._get_attribute_value(atm, self.private_name)

        if value is None:
            value = self.compute_value(atm)
            setattr(atm, self.private_name, value)
        return atm.return_value(value)

    def __set__(self, atm, value):
        self._reset_speeds(atm)
        if value is None:
            setattr(atm, self.private_name, None)
        else:
            setattr(atm, self.private_name, np.asarray(value))

    def _reset_speeds(self, atm):
        """To be used before setting a new speed value as private attribute."""
        for speed_attr in self._speed_attributes:
            setattr(atm, "_" + speed_attr, None)

    @staticmethod
    def _get_attribute_value(instance, attr_name):
        """
        :param instance:
        :param attr_name: name of desired attribute in provided instance
        :return: the value of the desired attribute, or None if the attribute does not exist
        """

        return instance.__dict__.get(attr_name)

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
