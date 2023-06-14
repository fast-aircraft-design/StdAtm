"""Functions for computation of atmosphere speed parameters."""
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

from functools import singledispatch
from numbers import Number

import numpy as np
from scipy.optimize import fsolve


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


# TRUE AIRSPEED =================================================================
# From Mach -----------------------------------------------------------------
def compute_tas_from_mach(mach, speed_of_sound):
    """

    :param mach: no unit
    :param speed_of_sound: in m/s
    :return: true airspeed in m/s
    """
    true_airspeed = mach * speed_of_sound
    return true_airspeed


# From EAS -----------------------------------------------------------------
def compute_tas_from_eas(equivalent_airspeed, density, sea_level_density):
    """

    :param equivalent_airspeed: in m/s
    :param density: in kg/m**3
    :param sea_level_density: in kg/m**3
    :return: true airspeed in m/s
    """
    true_airspeed = equivalent_airspeed * np.sqrt(sea_level_density / density)
    return true_airspeed


# From unitary Reynolds -----------------------------------------------------------------
def compute_tas_from_unit_re(unitary_reynolds, kinematic_viscosity):
    """

    :param unitary_reynolds: in 1/m
    :param kinematic_viscosity: in m**2/s
    :return: true airspeed in m/s
    """
    true_airspeed = unitary_reynolds * kinematic_viscosity
    return true_airspeed


# From dynamic pressure -----------------------------------------------------------------
def compute_tas_from_pdyn(dynamic_pressure, pressure, speed_of_sound):
    """

    :param dynamic_pressure: in Pa
    :param pressure: in Pa
    :param speed_of_sound: in m/s
    :return: true airspeed in m/s
    """
    true_airspeed = np.sqrt(dynamic_pressure / 0.7 / pressure) * speed_of_sound
    return true_airspeed


# MACH =================================================================
# From true airspeed -----------------------------------------------------------------
def compute_mach(true_airspeed, speed_of_sound):
    """

    :param true_airspeed: in m/s
    :param speed_of_sound: in m/s
    :return: Mach number (no unit)
    """
    mach = true_airspeed / speed_of_sound
    return mach


# From dynamic pressure -----------------------------------------------------------------
def compute_mach_from_pdyn(dynamic_pressure, pressure):
    """

    :param dynamic_pressure: in Pa
    :param pressure: in Pa
    :return: Mach number (no unit)
    """
    mach = np.sqrt(dynamic_pressure / 0.7 / pressure)
    return mach


# EQUIVALENT AIRSPEED =================================================================
def compute_equivalent_airspeed(true_airspeed, density, sea_level_density):
    """

    :param true_airspeed: in m/s
    :param density: in kg/m**3
    :param sea_level_density: in kg/m**3
    :return: equivalent airspeed in m/s
    """
    equivalent_airspeed = true_airspeed / np.sqrt(sea_level_density / density)
    return equivalent_airspeed


# UNITARY_REYNOLDS =================================================================
def compute_unitary_reynolds(true_airspeed, kinematic_viscosity):
    """

    :param true_airspeed: in m/s
    :param kinematic_viscosity: in m**2/s
    :return: unitary_reynolds in 1/m
    """
    unitary_reynolds = true_airspeed / kinematic_viscosity
    return unitary_reynolds


# DYNAMIC PRESSURE =================================================================
def compute_dynamic_pressure(mach, pressure):
    """

    :param mach: no unit
    :param pressure: in Pa
    :return: dynamic pressure in Pa
    """
    true_airspeed = 0.7 * mach**2 * pressure
    return true_airspeed


# IMPACT PRESSURE =================================================================
def _compute_subsonic_impact_pressure(mach, pressure):
    return pressure * ((1.0 + 0.2 * mach**2) ** 3.5 - 1.0)


def _compute_supersonic_impact_pressure(mach, pressure):
    # Rayleigh law
    # https://en.wikipedia.org/wiki/Rayleigh_flow#Additional_Rayleigh_Flow_Relations
    return pressure * (166.92158 * mach**7 / (7 * mach**2 - 1.0) ** 2.5 - 1.0)


@singledispatch
def compute_impact_pressure(mach, pressure):
    """

    :param mach: no unit
    :param pressure: in Pa
    :return: impact pressure in Pa
    """
    # Implementation for numpy arrays
    mach = np.asarray(mach)
    idx_subsonic = mach <= 1.0
    idx_supersonic = np.logical_not(idx_subsonic)

    if np.shape(pressure) != np.shape(mach):
        pressure = np.broadcast_to(pressure, np.shape(mach))
    else:
        pressure = np.asarray(pressure)

    impact_pressure = np.empty_like(mach)
    impact_pressure[idx_subsonic] = _compute_subsonic_impact_pressure(
        mach[idx_subsonic], pressure[idx_subsonic]
    )
    impact_pressure[idx_supersonic] = _compute_supersonic_impact_pressure(
        mach[idx_supersonic], pressure[idx_supersonic]
    )
    return impact_pressure


@compute_impact_pressure.register
def _(mach: Number, pressure: Number):
    # Implementation for floats
    if mach <= 1.0:
        return _compute_subsonic_impact_pressure(mach, pressure)
    else:
        return _compute_supersonic_impact_pressure(mach, pressure)


# CALIBRATED AIRSPEED =================================================================


#         Computation is done using Eq. 3.16 and 3.17 from:
#         Gracey, William (1980), "Measurement of Aircraft Speed and Altitude",
#         NASA Reference Publication 1046.
#         https://apps.dtic.mil/sti/pdfs/ADA280006.pdf


def _compute_cas_low_speed(impact_pressure, sea_level_pressure, sea_level_speed_of_sound):
    return sea_level_speed_of_sound * np.sqrt(
        5.0 * ((impact_pressure / sea_level_pressure + 1.0) ** (1.0 / 3.5) - 1.0)
    )


def _compute_cas_high_speed(impact_pressure, sea_level_pressure, sea_level_speed_of_sound):
    def _equation_cas_high_speed(cas, impact_pressure):
        return cas - sea_level_speed_of_sound * (
            (impact_pressure / sea_level_pressure + 1)
            * (7 * (cas / sea_level_speed_of_sound) ** 2 - 1) ** 2.5
            / (6**2.5 * 1.2**3.5)
        ) ** (1 / 7)

    root = fsolve(
        _equation_cas_high_speed,
        x0=sea_level_speed_of_sound * np.ones_like(impact_pressure),
        args=(impact_pressure,),
    )
    return root


def compute_calibrated_airspeed(impact_pressure, sea_level_pressure, sea_level_speed_of_sound):
    impact_pressure = np.asarray(impact_pressure)

    calibrated_airspeed = np.asarray(
        _compute_cas_low_speed(impact_pressure, sea_level_pressure, sea_level_speed_of_sound)
    )

    idx_high_speed = calibrated_airspeed > sea_level_speed_of_sound
    if np.any(idx_high_speed):
        calibrated_airspeed[idx_high_speed] = _compute_cas_high_speed(
            impact_pressure[idx_high_speed], sea_level_pressure, sea_level_speed_of_sound
        )

    return calibrated_airspeed
