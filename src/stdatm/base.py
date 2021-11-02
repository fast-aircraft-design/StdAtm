"""Base classes for atmosphere calculations."""
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

from abc import ABC, abstractmethod
from copy import copy

import numpy as np
from scipy.optimize import fsolve


class AbstractParameterCalculator(ABC):
    """
    Abstract class for computing parameter values.

    A subclass has to implement :meth:`compute_value` that will return the value
    of current parameter from other data in an Atmosphere instance.

    This class provides also the method :meth:`compute_base_parameter` that will
    compute the "base parameter" from current parameter (and possibly
    other data in an Atmosphere instance). The default implementation iteratively
    solves :code:`f(base_parameter) = current_parameter` and should be replaced
    by an implementation of :code:`g(current_parameter) = base_parameter` when possible.

    The base parameter is defined when subclassing this class, like ::

        class NewParameter(AbstractParameterCalculator, base_parameter="altitude")

    """

    base_parameter: str

    @classmethod
    def __init_subclass__(cls, *, base_parameter: str = None):
        """

        :param base_parameter: the base parameter that is central to computations
        """
        if base_parameter:
            cls.base_parameter = base_parameter

    @abstractmethod
    def compute_value(self, atm):
        """
        Implement here the calculation of the current parameter.

        :param atm: the parent Atmosphere instance
        :return: value of the parameter, computed from available data in atm
        """

    def compute_base_parameter(self, atm, value):
        """
        Computes base parameter from parameter value.

        This method provides a default implementation that iteratively solves the problem
        using :meth:`compute_value`.

        You may overload this method to provide a direct method.

        :param atm: the parent Atmosphere instance
        :param value: value of the current parameter
        :return: value of base parameter
        """

        solver_atm = copy(atm)
        shape = np.shape(value)
        value = np.ravel(value)
        root = fsolve(
            lambda x: value - self._compute_parameter(x, solver_atm, shape),
            x0=500.0 * np.ones_like(value),
        )
        return np.reshape(root, shape)

    def _compute_parameter(self, base_parameter_value, atm, shape):
        """

        :param base_parameter_value:
        :param atm:
        :param shape:
        :return:
        """
        setattr(atm, self.base_parameter, np.reshape(base_parameter_value, shape))
        return np.ravel(self.compute_value(atm))


class AbstractStaticCalculator(AbstractParameterCalculator, ABC, base_parameter="altitude"):
    """Abstract class for computing static parameter values."""

    def compute_altitude(self, atm, value):
        """
        Computes altitude from parameter value.

        This method provides a default implementation that iteratively solves the problem
        using :meth:`compute_value`.

        You may overload this method to provide a direct method.

        :param atm: the parent Atmosphere instance
        :param value: value of the current static parameter
        :return: value of altitude in m
        """
        return self.compute_base_parameter(atm, value)


class AbstractSpeedParameter(AbstractParameterCalculator, ABC, base_parameter="true_airspeed"):
    """Abstract class for computing speed values."""

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
        return self.compute_base_parameter(atm, value)


class UnsettableAtmosphereParameter:
    """
    Descriptor class for unsettable parameters in
    :class:`~fastoad.model_base.atmosphere.atmosphere.Atmosphere`.

    Such parameters cannot be set, but are computed when their value is requested.
    """

    #: This dict will associate descriptor name (in Atmosphere class) to the descriptor class.
    #: It is useful for doing operations on all parameters.
    parameter_attributes = {}

    def __init__(self, parameter_calculator: AbstractParameterCalculator):
        self._parameter_calculator = parameter_calculator
        self.__doc__ = self._parameter_calculator.__doc__  # For correct documentation in Sphinx

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name
        self.parameter_attributes[name] = type(self._parameter_calculator)

        setattr(owner, self.private_name, None)

    def __get__(self, atm, owner):
        value = getattr(atm, self.private_name)

        if value is None:
            value = self._parameter_calculator.compute_value(atm)
            setattr(atm, self.private_name, value)
        return atm.return_value(value)

    @classmethod
    def reset_parameters(cls, atm):
        """
        Resets all related parameters to None.

        To be used before setting a parameter value as private attribute to
        ensure they will be computed again the next time they are used.
        """
        for attr in cls.parameter_attributes:
            setattr(atm, "_" + attr, None)

    @staticmethod
    def get_attribute_value(instance, attr_name):
        """
        :param instance:
        :param attr_name: name of desired attribute in provided instance
        :return: the value of the desired attribute, or None if the attribute does not exist
        """
        return instance.__dict__.get(attr_name)


class AtmosphereParameter(UnsettableAtmosphereParameter):
    """
    Descriptor class for settable parameters in
    :class:`~fastoad.model_base.atmosphere.atmosphere.Atmosphere`.
    """

    def __set__(self, atm, value):
        self.reset_parameters(atm)
        if value is not None:
            # Note: it's important to specify dtype. In some cases, having integers
            # as input will lead to an int array that may cause problems later.
            value = np.asarray(value, dtype=np.float64)
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


# The next two classes are created so that each one has its own class attribute
# parameter_attributes.
class StaticParameter(AtmosphereParameter):
    """
    Descriptor class for static parameters in
    :class:`~fastoad.model_base.atmosphere.atmosphere.Atmosphere`.
    """

    @classmethod
    def reset_parameters(cls, atm):
        # If a static parameter is changed, speed parameters should be reset as well.
        super().reset_parameters(atm)
        SpeedParameter.reset_parameters(atm)


class SpeedParameter(AtmosphereParameter):
    """
    Descriptor class for speed parameters in
    :class:`~fastoad.model_base.atmosphere.atmosphere.Atmosphere`.
    """
