"""Tests for Atmosphere class"""
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
import pytest
from numpy.testing import assert_allclose
from scipy.constants import foot

from .. import AtmosphereSI
from ..atmosphere import Atmosphere, AtmosphereSI


def test_atmosphere():
    """Tests properties of Atmosphere class."""
    # Altitudes in meters Values at disa=0 from "Advanced Aircraft Design (
    # Egbert TORENBEEK, Oxford, UK: John Wiley & Sons Ltd, 2013) Appendix B,
    # p.397-398" Values at disa=10 from
    # https://www.digitaldutch.com/atmoscalc/, with a 0.98749 factor on
    # viscosity because at sea level and disa=0, the calculator gives
    # 1.81206e-5 for dynamic viscosity, though ISA assumption is 1.7894e-5
    expectations = np.array(
        [
            (0, 0, 288.15, 1.225, 101325, 1.460e-05, 340.29),
            (500, 0, 284.90, 1.1673, 95461, 1.519e-05, 338.37),
            (1000, 0, 281.65, 1.1117, 89874, 1.581e-05, 336.43),
            (1500, 0, 278.40, 1.0581, 84556, 1.646e-05, 334.49),
            (2000, 0, 275.15, 1.0065, 79495, 1.714e-05, 332.53),
            (2500, 0, 271.90, 0.9569, 74682, 1.787e-05, 330.56),
            (3000, 0, 268.65, 0.9091, 70108, 1.863e-05, 328.58),
            (3500, 0, 265.40, 0.8632, 65764, 1.943e-05, 326.58),
            (4000, 0, 262.15, 0.8191, 61640, 2.028e-05, 324.58),
            (4500, 0, 258.90, 0.7768, 57728, 2.117e-05, 322.56),
            (5000, 0, 255.65, 0.7361, 54020, 2.211e-05, 320.53),
            (5500, 0, 252.40, 0.6971, 50506, 2.311e-05, 318.48),
            (6000, 0, 249.15, 0.6597, 47181, 2.417e-05, 316.43),
            (6500, 0, 245.90, 0.6238, 44034, 2.529e-05, 314.36),
            (7000, 0, 242.65, 0.5895, 41060, 2.648e-05, 312.27),
            (7500, 0, 239.40, 0.5566, 38251, 2.773e-05, 310.17),
            (8000, 0, 236.15, 0.5252, 35599, 2.906e-05, 308.06),
            (8500, 0, 232.90, 0.4951, 33099, 3.048e-05, 305.93),
            (9000, 0, 229.65, 0.4663, 30742, 3.199e-05, 303.79),
            (9500, 0, 226.40, 0.4389, 28523, 3.359e-05, 301.63),
            (10000, 0, 223.15, 0.4127, 26436, 3.530e-05, 299.46),
            (10500, 0, 219.90, 0.3877, 24474, 3.712e-05, 297.27),
            (11000, 0, 216.65, 0.3639, 22632, 3.905e-05, 295.07),
            (12000, 0, 216.65, 0.3108, 19330, 4.573e-05, 295.07),
            (13000, 0, 216.65, 0.2655, 16510, 5.353e-05, 295.07),
            (14000, 0, 216.65, 0.2268, 14101, 6.266e-05, 295.07),
            (15000, 0, 216.65, 0.1937, 12044, 7.337e-05, 295.07),
            (16000, 0, 216.65, 0.1654, 10287, 8.592e-05, 295.07),
            (17000, 0, 216.65, 0.1413, 8786, 1.006e-04, 295.07),
            (18000, 0, 216.65, 0.1207, 7505, 1.177e-04, 295.07),
            (19000, 0, 216.65, 0.1031, 6410, 1.378e-04, 295.07),
            (20000, 0, 216.65, 0.088, 5475, 1.615e-04, 295.07),
            (0, 10, 298.15, 1.1839, 101325, 1.5527e-5, 346.15),
            (1000, 10, 291.65, 1.0735, 89875, 1.6829e-5, 342.36),
            (3000, 10, 278.65, 0.87650, 70108, 1.9877e-5, 334.64),
            (10000, 10, 233.15, 0.39500, 26436, 3.8106e-05, 306.10),
            (14000, 10, 226.65, 0.2167, 14102, 6.7808e-05, 301.80),
        ],
        dtype=[
            ("alt", "i8"),
            ("dT", "f4"),
            ("T", "f4"),
            ("rho", "f4"),
            ("P", "f4"),
            ("visc", "f4"),
            ("SoS", "f4"),
        ],
    )

    for values in expectations:
        # Checking with altitude provided as scalar
        # Using AtmosphereSI allows also to test having an integer as input.
        alt = values["alt"]
        assert isinstance(alt, (int, np.integer))
        atm = AtmosphereSI(alt, values["dT"])
        assert values["T"] == pytest.approx(atm.temperature, rel=1e-4)
        assert values["rho"] == pytest.approx(atm.density, rel=1e-3)
        assert values["P"] == pytest.approx(atm.pressure, rel=1e-4)
        assert values["visc"] == pytest.approx(atm.kinematic_viscosity, rel=1e-2)
        assert values["SoS"] == pytest.approx(atm.speed_of_sound, rel=1e-3)

        # Checking with altitude provided as one-element list
        alt = [values["alt"] / foot]
        assert isinstance(alt, list)
        atm = Atmosphere(alt, values["dT"])
        assert values["T"] == pytest.approx(atm.temperature, rel=1e-4)
        assert values["rho"] == pytest.approx(atm.density, rel=1e-3)
        assert values["P"] == pytest.approx(atm.pressure, rel=1e-4)
        assert values["visc"] == pytest.approx(atm.kinematic_viscosity, rel=1e-2)
        assert values["SoS"] == pytest.approx(atm.speed_of_sound, rel=1e-3)

    for delta_t in [0, 10]:
        idx = expectations["dT"] == delta_t

        # Checking with altitude provided as 1D numpy array
        alt = expectations["alt"][idx] / foot
        assert isinstance(alt, np.ndarray)
        assert len(alt.shape) == 1
        atm = Atmosphere(alt, delta_t)
        assert expectations["T"][idx] == pytest.approx(atm.temperature, rel=1e-4)
        assert expectations["rho"][idx] == pytest.approx(atm.density, rel=1e-3)
        assert expectations["P"][idx] == pytest.approx(atm.pressure, rel=1e-4)
        assert expectations["visc"][idx] == pytest.approx(atm.kinematic_viscosity, rel=1e-2)
        assert expectations["SoS"][idx] == pytest.approx(atm.speed_of_sound, rel=1e-3)
        # Additional check for get_altitude in meters
        assert expectations["alt"][idx] == pytest.approx(
            atm.get_altitude(altitude_in_feet=False), rel=1e-3
        )

        # Checking with altitude provided as a list and in meters
        alt = expectations["alt"][idx].tolist()
        assert isinstance(alt, list)
        atm = Atmosphere(alt, delta_t, altitude_in_feet=False)
        assert expectations["T"][idx] == pytest.approx(atm.temperature, rel=1e-4)
        assert expectations["rho"][idx] == pytest.approx(atm.density, rel=1e-3)
        assert expectations["P"][idx] == pytest.approx(atm.pressure, rel=1e-4)
        assert expectations["visc"][idx] == pytest.approx(atm.kinematic_viscosity, rel=1e-2)
        assert expectations["SoS"][idx] == pytest.approx(atm.speed_of_sound, rel=1e-3)
        # Additional check for get_altitude in feet
        assert expectations["alt"][idx] / foot == pytest.approx(atm.get_altitude(), rel=1e-3)

        # Same with AtmosphereSI
        atm = AtmosphereSI(alt, delta_t)
        assert expectations["T"][idx] == pytest.approx(atm.temperature, rel=1e-4)
        assert expectations["rho"][idx] == pytest.approx(atm.density, rel=1e-3)
        assert expectations["P"][idx] == pytest.approx(atm.pressure, rel=1e-4)
        assert expectations["visc"][idx] == pytest.approx(atm.kinematic_viscosity, rel=1e-2)
        assert expectations["SoS"][idx] == pytest.approx(atm.speed_of_sound, rel=1e-3)
        # Additional check for altitude property
        assert expectations["alt"][idx] == pytest.approx(atm.altitude, rel=1e-3)


def test_speed_parameters_basic():
    atm = Atmosphere([0, 5000, 10000])
    with pytest.raises(RuntimeError):
        atm.true_airspeed = [[100, 200]]


def test_speed_conversions_with_broadcast():
    """Tests for speed conversions with different but compatible shapes for altitude and TAS"""
    run_speed_conversion_tests(with_broadcast=True)


def test_speed_conversions_without_broadcast():
    """Tests for speed conversions with identical shapes for altitude and TAS"""
    run_speed_conversion_tests(with_broadcast=False)


def run_speed_conversion_tests(with_broadcast: bool):
    if with_broadcast:
        altitudes = [0.0, 1000.0, 35000.0]
        TAS = [
            [100.0],
            [200.0],
            [270.0],
            [300.0],
            [400.0],
            [800.0],
        ]
        speed_of_sound = [340.294, 339.122, 296.536]
    else:
        altitudes = [
            [0.0, 1000.0, 35000.0],
            [0.0, 1000.0, 35000.0],
            [0.0, 1000.0, 35000.0],
            [0.0, 1000.0, 35000.0],
            [0.0, 1000.0, 35000.0],
            [0.0, 1000.0, 35000.0],
        ]
        TAS = [
            [100.0, 100.0, 100.0],
            [200.0, 200.0, 200.0],
            [270.0, 270.0, 270.0],
            [300.0, 300.0, 300.0],
            [400.0, 400.0, 400.0],
            [800.0, 800.0, 800.0],
        ]
        speed_of_sound = [
            [340.294, 339.122, 296.536],
            [340.294, 339.122, 296.536],
            [340.294, 339.122, 296.536],
            [340.294, 339.122, 296.536],
            [340.294, 339.122, 296.536],
            [340.294, 339.122, 296.536],
        ]

    atm = Atmosphere(altitudes)
    assert atm.true_airspeed is None
    assert atm.equivalent_airspeed is None
    assert atm.mach is None
    assert atm.unitary_reynolds is None

    atm.true_airspeed = TAS
    Checker.check_speeds(atm)

    atm.true_airspeed = None
    assert atm.true_airspeed is None
    assert atm.equivalent_airspeed is None
    assert atm.mach is None
    assert atm.unitary_reynolds is None

    atm = Atmosphere(altitudes)
    atm.equivalent_airspeed = Checker.expected_EAS
    Checker.check_speeds(atm)

    # Here we do not instantiate a new Atmosphere, but simply modify Mach number.
    # Other parameters should be modified accordingly.
    atm.mach = 1.0
    assert_allclose(atm.true_airspeed, speed_of_sound, rtol=1e-4)

    atm = Atmosphere(altitudes)
    atm.mach = Checker.expected_Mach
    Checker.check_speeds(atm)

    atm = Atmosphere(altitudes)
    atm.unitary_reynolds = Checker.expected_Re1
    Checker.check_speeds(atm)

    atm = Atmosphere(altitudes)
    atm.dynamic_pressure = Checker.expected_dynamic_pressure
    Checker.check_speeds(atm)

    atm = Atmosphere(altitudes)
    atm.impact_pressure = Checker.expected_impact_pressure
    Checker.check_speeds(atm)

    # Check with one altitude value, but several speed values ############################
    atm = Atmosphere(35000)
    atm.true_airspeed = np.array(Checker.expected_TAS)[:, 2]
    assert_allclose(atm.equivalent_airspeed, np.array(Checker.expected_EAS)[:, 2], rtol=1e-4)
    assert_allclose(atm.mach, np.array(Checker.expected_Mach)[:, 2], rtol=1e-4)
    assert_allclose(atm.unitary_reynolds, np.array(Checker.expected_Re1)[:, 2], rtol=1e-4)
    assert_allclose(
        atm.impact_pressure, np.array(Checker.expected_impact_pressure)[:, 2], rtol=1e-4
    )


class Checker:
    # source:  https://www.newbyte.co.il/calculator/index.php using "pressure altitude" as output.
    # This source, with "geometric altitude" as input, agrees with
    # http://www.aerospaceweb.org/design/scripts/atmosphere/
    #
    # Warning: http://www.hochwarth.com/misc/AviationCalculator.html explicitly tells that
    # the speed conversion is valid only in subsonic domain.
    # https://aerotoolbox.com/airspeed-conversions/ has the same limitation, though it is not
    # explicitly written.

    expected_TAS = [
        [100.0, 100.0, 100.0],
        [200.0, 200.0, 200.0],
        [270.0, 270.0, 270.0],
        [300.0, 300.0, 300.0],
        [400.0, 400.0, 400.0],
        [800.0, 800.0, 800.0],
    ]
    expected_EAS = [
        [100.0, 98.543, 55.666],
        [200.0, 197.085, 111.333],
        [270, 266.065, 150.299],
        [300.0, 295.628, 166.999],
        [400, 394.170, 222.666],
        [800, 788.341, 445.332],
    ]
    expected_CAS = [
        [100.0, 98.580, 56.269],
        [200.0, 197.362, 116.073],
        [270, 266.698, 161.732],
        [300.0, 296.465, 182.507],
        [400, 395.578, 252.396],
        [800, 789.350, 479.567],
    ]
    expected_Mach = [
        [0.29386, 0.29488, 0.33723],
        [0.58773, 0.58976, 0.67446],
        [0.79343, 0.79617, 0.91051],
        [0.88159, 0.88464, 1.01168],
        [1.17545, 1.17952, 1.34891],
        [2.35091, 2.35903, 2.69782],
    ]
    expected_Re1 = [
        [6845941, 6683613, 2648139],
        [13691882, 13367227, 5296278],
        [18484040, 18045756, 7149975],
        [20537823, 20050840, 7944417],
        [27383763, 26734454, 10592556],
        [54767527, 53468908, 21185111],
    ]
    expected_dynamic_pressure = [
        [6125.0, 5947.8, 1898.0],
        [24500.0, 23791.1, 7591.9],
        [44651.2, 43359.2, 13836.3],
        [55125.0, 53529.9, 17081.9],
        [97999.9, 95164.2, 30367.8],
        [391999.7, 380656.9, 121471.0],
    ]
    expected_impact_pressure = [
        [6258.4, 6078.2, 1952.6],
        [26689.4, 25932.3, 8495.0],
        [52127.8, 50672.8, 16946.6],
        [66684.1, 64838.1, 21911.2],
        [135479.4, 131780.5, 44684.1],
        [668493.9, 649486.1, 210939.7],
    ]

    @classmethod
    def check_speeds(cls, atm, tol=1e-4):

        assert_allclose(atm.true_airspeed, cls.expected_TAS, rtol=tol)
        assert_allclose(atm.equivalent_airspeed, cls.expected_EAS, rtol=tol)
        assert_allclose(atm.calibrated_airspeed, cls.expected_CAS, rtol=tol)
        assert_allclose(atm.mach, cls.expected_Mach, rtol=tol)
        assert_allclose(atm.unitary_reynolds, cls.expected_Re1, rtol=tol)
        assert_allclose(atm.dynamic_pressure, cls.expected_dynamic_pressure, rtol=tol)
        assert_allclose(atm.impact_pressure, cls.expected_impact_pressure, rtol=tol)


@pytest.fixture(scope="session")
def altitude():
    return np.linspace(0.0, 20000.0, int(5e7))


@pytest.fixture(scope="session")
def atmosphere1(altitude):
    atm = AtmosphereSI(altitude)
    atm.true_airspeed = 200.0
    return atm


def test_performances_temperature_array(atmosphere1):
    _ = atmosphere1.temperature


def test_performances_pressure_array(atmosphere1):
    _ = atmosphere1.pressure


def test_performances_density_array(atmosphere1):
    _ = atmosphere1.density


def test_performances_kinematic_viscosity_array(atmosphere1):
    _ = atmosphere1.kinematic_viscosity


def test_performances_speed_of_sound_array(atmosphere1):
    _ = atmosphere1.speed_of_sound


def test_performances_TAS_1_array(atmosphere1):
    _ = atmosphere1.true_airspeed


def test_performances_EAS_1_array(atmosphere1):
    _ = atmosphere1.equivalent_airspeed


def test_performances_mach_1_array(atmosphere1):
    _ = atmosphere1.mach


def test_performances_unit_Re_1_array(atmosphere1):
    _ = atmosphere1.unitary_reynolds


def test_performances_TAS_2_array(atmosphere1):
    atmosphere1.true_airspeed = 500.0
    _ = atmosphere1.true_airspeed


def test_performances_EAS_2_array(atmosphere1):
    _ = atmosphere1.equivalent_airspeed


def test_performances_mach_2_array(atmosphere1):
    _ = atmosphere1.mach


def test_performances_unit_Re_2_array(atmosphere1):
    _ = atmosphere1.unitary_reynolds


def test_performances_reask_array(atmosphere1):
    _ = atmosphere1.temperature
    _ = atmosphere1.pressure
    _ = atmosphere1.density
    _ = atmosphere1.kinematic_viscosity
    _ = atmosphere1.speed_of_sound
    _ = atmosphere1.true_airspeed
    _ = atmosphere1.equivalent_airspeed
    _ = atmosphere1.mach
    _ = atmosphere1.unitary_reynolds


def test_performances_loop_static(altitude):
    for alt in altitude[::1000]:
        atm = AtmosphereSI(alt)
        _ = atm.temperature
        _ = atm.pressure
        _ = atm.density
        _ = atm.kinematic_viscosity
        _ = atm.speed_of_sound


def test_performances_loop_speeds_init_TAS(altitude):
    for alt in altitude[::1000]:
        atm = AtmosphereSI(alt)
        atm.true_airspeed = 100.0
        _ = atm.true_airspeed
        _ = atm.equivalent_airspeed
        _ = atm.mach
        _ = atm.unitary_reynolds


def test_performances_loop_speeds_init_mach(altitude):
    for alt in altitude[::1000]:
        atm = AtmosphereSI(alt)
        atm.mach = 0.3
        _ = atm.true_airspeed
        _ = atm.equivalent_airspeed
        _ = atm.mach
        _ = atm.unitary_reynolds
