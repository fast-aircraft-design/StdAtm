# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import foot

from ..atmosphere_partials import AtmosphereWithPartials
from ..atmosphere import Atmosphere


@pytest.fixture(scope="session")
def altitude():
    return np.linspace(0.0, 20000.0, int(1e6))


def get_atmosphere(altitude, altitude_in_feet):
    atm_part = AtmosphereWithPartials(altitude, 0.0, altitude_in_feet)
    return atm_part


def get_fd_partial(altitude, parameter_name, altitude_in_feet=False, step=1e-6):
    atm_minus_step = Atmosphere(altitude - step, altitude_in_feet=altitude_in_feet)
    atm_plus_step = Atmosphere(altitude + step, altitude_in_feet=altitude_in_feet)
    return (getattr(atm_plus_step, parameter_name) - getattr(atm_minus_step, parameter_name)) / (
        2.0 * step
    )


def test_performances_temperature_partials_array(altitude, benchmark):
    def func():
        atm = get_atmosphere(altitude, False)
        _ = atm.partial_temperature_altitude

    benchmark(func)


def test_temperature_partials_against_fd(altitude):
    atm = get_atmosphere(altitude, False)

    computed_partials = atm.partial_temperature_altitude
    verify_partials = get_fd_partial(altitude, "temperature")

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_temperature_partials_against_fd_ft(altitude):
    atm = get_atmosphere(altitude / foot, True)

    computed_partials = atm.partial_temperature_altitude
    verify_partials = get_fd_partial(altitude / foot, "temperature", True)

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_performances_pressure_partials_array(altitude, benchmark):
    def func():
        atm = get_atmosphere(altitude, False)
        _ = atm.partial_pressure_altitude

    benchmark(func)


def test_pressure_partials_against_fd(altitude):
    atm = get_atmosphere(altitude, False)

    computed_partials = atm.partial_pressure_altitude
    verify_partials = get_fd_partial(altitude, "pressure")

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_pressure_partials_against_fd_ft(altitude):
    atm = get_atmosphere(altitude / foot, True)

    computed_partials = atm.partial_pressure_altitude
    verify_partials = get_fd_partial(altitude / foot, "pressure", True)

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_performances_density_partials_array(altitude, benchmark):
    def func():
        atm = get_atmosphere(altitude, False)
        _ = atm.partial_density_altitude

    benchmark(func)


def test_density_partials_against_fd(altitude):
    atm = get_atmosphere(altitude, False)

    computed_partials = atm.partial_density_altitude
    verify_partials = get_fd_partial(altitude, "density")

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_density_partials_against_fd_ft(altitude):
    atm = get_atmosphere(altitude / foot, True)

    computed_partials = atm.partial_density_altitude
    verify_partials = get_fd_partial(altitude / foot, "density", True)

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_performances_speed_of_sound_partials_array(altitude, benchmark):
    def func():
        atm = get_atmosphere(altitude, False)
        _ = atm.partial_speed_of_sound_altitude

    benchmark(func)


def test_speed_of_sound_partials_against_fd(altitude):
    atm = get_atmosphere(altitude, False)

    computed_partials = atm.partial_speed_of_sound_altitude
    verify_partials = get_fd_partial(altitude, "speed_of_sound")

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_speed_of_sound_partials_against_fd_ft(altitude):
    atm = get_atmosphere(altitude / foot, True)

    computed_partials = atm.partial_speed_of_sound_altitude
    verify_partials = get_fd_partial(altitude / foot, "speed_of_sound", True)

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_performances_dynamic_viscosity_partials_array(altitude, benchmark):
    def func():
        atm = get_atmosphere(altitude, False)
        _ = atm.partial_dynamic_viscosity_altitude

    benchmark(func)


def test_dynamic_viscosity_partials_against_fd(altitude):
    atm = get_atmosphere(altitude, False)

    computed_partials = atm.partial_dynamic_viscosity_altitude
    verify_partials = get_fd_partial(altitude, "dynamic_viscosity")

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_dynamic_viscosity_partials_against_fd_ft(altitude):
    atm = get_atmosphere(altitude / foot, True)

    computed_partials = atm.partial_dynamic_viscosity_altitude
    verify_partials = get_fd_partial(altitude / foot, "dynamic_viscosity", True)

    assert_allclose(computed_partials, verify_partials, rtol=1e-4)


def test_performances_kinematic_viscosity_partials_array(altitude, benchmark):
    def func():
        atm = get_atmosphere(altitude, False)
        _ = atm.partial_kinematic_viscosity_altitude

    benchmark(func)


def test_kinematic_viscosity_partials_against_fd(altitude):
    atm = get_atmosphere(altitude, False)

    computed_partials = atm.partial_kinematic_viscosity_altitude
    verify_partials = get_fd_partial(altitude, "kinematic_viscosity")

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_kinematic_viscosity_partials_against_fd_ft(altitude):
    atm = get_atmosphere(altitude / foot, True)

    computed_partials = atm.partial_kinematic_viscosity_altitude
    verify_partials = get_fd_partial(altitude / foot, "kinematic_viscosity", True)

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_performances_reask_array(altitude, benchmark):
    atm = get_atmosphere(altitude, False)
    _ = atm.partial_temperature_altitude
    _ = atm.partial_pressure_altitude
    _ = atm.partial_density_altitude
    _ = atm.partial_speed_of_sound_altitude
    _ = atm.partial_dynamic_viscosity_altitude
    _ = atm.partial_kinematic_viscosity_altitude

    def func():
        _ = atm.partial_temperature_altitude
        _ = atm.partial_pressure_altitude
        _ = atm.partial_density_altitude
        _ = atm.partial_speed_of_sound_altitude
        _ = atm.partial_dynamic_viscosity_altitude
        _ = atm.partial_kinematic_viscosity_altitude

    benchmark(func)
