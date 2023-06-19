"""
Implementation of International Standard Atmosphere with partial derivatives of state parameters
with respect to altitude.
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

from typing import Sequence, Union

import numpy as np
from scipy.constants import R, atmosphere, foot

from .atmosphere import Atmosphere

from .partials_state_parameters import (
    compute_partial_temperature,
    compute_partial_pressure,
    compute_partial_density,
    compute_partial_speed_of_sound,
)

AIR_MOLAR_MASS = 28.9647e-3
AIR_GAS_CONSTANT = R / AIR_MOLAR_MASS
SEA_LEVEL_PRESSURE = atmosphere
SEA_LEVEL_TEMPERATURE = 288.15
TROPOPAUSE = 11000


class AtmospherePartials(Atmosphere):
    """
    Implementation of International Standard Atmosphere for troposphere and stratosphere with
    derivatives of state parameters with respect to altitude.

    Atmosphere properties and partials are provided in the same "shape" as provided
    altitude:

    - if altitude is given as a float, returned values will be floats
    - if altitude is given as a sequence (list, 1D numpy array, ...), returned
      values will be 1D numpy arrays
    - if altitude is given as nD numpy array, returned values will be nD numpy
      arrays

    Usage:

    TO BE FILLED
    """

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

        # Can't use **kwargs because we need to define unit_coeff as a an attribute for the
        # computation of partials
        super().__init__(altitude=altitude, delta_t=delta_t, altitude_in_feet=altitude_in_feet)

        self._unit_coeff = foot if altitude_in_feet else 1.0

        # Partials
        self._partials_temperature_altitude = None
        self._partials_pressure_altitude = None
        self._partials_density_altitude = None
        self._partials_speed_of_sound_altitude = None
        self._partials_dynamic_viscosity_altitude = None
        self._partials_kinematic_viscosity_altitude = None

    @property
    def partial_temperature_altitude(self) -> Union[float, np.ndarray]:
        """
        Partial derivative of the temperature in K with respect to the altitude in the unit
        provided.
        """

        if self._partials_temperature_altitude is None:
            self._partials_temperature_altitude = compute_partial_temperature(
                self._altitude, self._unit_coeff
            )

        return self._partials_temperature_altitude

    @property
    def partial_pressure_altitude(self) -> Union[float, np.ndarray]:
        """
        Partial derivative of the pressure in Pa with respect to the altitude in the unit
        provided.
        """

        if self._partials_pressure_altitude is None:
            self._partials_pressure_altitude = compute_partial_pressure(
                self._altitude, self._unit_coeff
            )

        return self._partials_pressure_altitude

    @property
    def partial_density_altitude(self) -> Union[float, np.ndarray]:
        """
        Partial derivative of the density in kg/m**3 with respect to the altitude in the unit
        provided.
        """

        if self._partials_density_altitude is None:
            self._partials_density_altitude = compute_partial_density(
                self.temperature,
                self.pressure,
                self.partial_temperature_altitude,
                self.partial_pressure_altitude,
            )

        return self._partials_density_altitude

    @property
    def partial_speed_of_sound_altitude(self) -> Union[float, np.ndarray]:
        """
        Partial derivative of the speed of sound in m/s with respect to the altitude in the unit
        provided.
        """

        if self._partials_speed_of_sound_altitude is None:
            self._partials_speed_of_sound_altitude = compute_partial_speed_of_sound(
                self.temperature,
                self.partial_temperature_altitude,
            )

        return self._partials_speed_of_sound_altitude
