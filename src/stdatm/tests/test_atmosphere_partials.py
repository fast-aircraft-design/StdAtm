# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import foot

from ..atmosphere_partials import AtmospherePartials
from ..atmosphere import Atmosphere

STEP = 1e-6


@pytest.fixture(scope="session")
def altitude():
    return np.linspace(0.0, 20000.0, int(1e6))


def get_atmosphere(altitude, altitude_in_feet):
    atm_part = AtmospherePartials(altitude, 0.0, altitude_in_feet)
    return atm_part


def get_atmosphere_minus_step(altitude):
    atm = Atmosphere(altitude - STEP, altitude_in_feet=False)
    return atm


def get_atmosphere_plus_step(altitude):
    atm = Atmosphere(altitude + STEP, altitude_in_feet=False)
    return atm


def get_atmosphere_minus_step_ft(altitude):
    atm = Atmosphere(altitude / foot - STEP, altitude_in_feet=True)
    return atm


def get_atmosphere_plus_step_ft(altitude):
    atm = Atmosphere(altitude / foot + STEP, altitude_in_feet=True)
    return atm


def test_performances_temperature_partials_array(altitude, benchmark):
    def func():
        atm = get_atmosphere(altitude, False)
        _ = atm.partial_temperature_altitude

    benchmark(func)


def test_temperature_partials_against_fd(altitude):
    atm_minus = get_atmosphere_minus_step(altitude)
    atm_plus = get_atmosphere_plus_step(altitude)

    atm = get_atmosphere(altitude, False)

    computed_partials = atm.partial_temperature_altitude
    verify_partials = (atm_plus.temperature - atm_minus.temperature) / (2.0 * STEP)

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_temperature_partials_against_fd_ft(altitude):
    atm_minus = get_atmosphere_minus_step_ft(altitude)
    atm_plus = get_atmosphere_plus_step_ft(altitude)

    atm = get_atmosphere(altitude / foot, True)

    computed_partials = atm.partial_temperature_altitude
    verify_partials = (atm_plus.temperature - atm_minus.temperature) / (2.0 * STEP)

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_performances_pressure_partials_array(altitude, benchmark):
    def func():
        atm = get_atmosphere(altitude, False)
        _ = atm.partial_pressure_altitude

    benchmark(func)


def test_pressure_partials_against_fd(altitude):
    atm_minus = get_atmosphere_minus_step(altitude)
    atm_plus = get_atmosphere_plus_step(altitude)

    atm = get_atmosphere(altitude, False)

    computed_partials = atm.partial_pressure_altitude
    verify_partials = (atm_plus.pressure - atm_minus.pressure) / (2.0 * STEP)

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_pressure_partials_against_fd_ft(altitude):
    atm_minus = get_atmosphere_minus_step_ft(altitude)
    atm_plus = get_atmosphere_plus_step_ft(altitude)

    atm = get_atmosphere(altitude / foot, True)

    computed_partials = atm.partial_pressure_altitude
    verify_partials = (atm_plus.pressure - atm_minus.pressure) / (2.0 * STEP)

    assert_allclose(computed_partials, verify_partials, rtol=5e-5)


def test_performances_reask_array(altitude, benchmark):
    atm = get_atmosphere(altitude, False)
    _ = atm.partial_temperature_altitude
    _ = atm.partial_pressure_altitude

    def func():
        _ = atm.partial_temperature_altitude
        _ = atm.partial_pressure_altitude

    benchmark(func)
