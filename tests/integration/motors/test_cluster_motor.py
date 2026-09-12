# pylint: disable=invalid-name
import pytest
import numpy as np
from rocketpy import SolidMotor, Function
from rocketpy.motors.cluster_motor import ClusterMotor


@pytest.fixture
def base_motor():
    """
    Creates a simplified SolidMotor for testing purposes.
    Properties:
    - Constant Thrust: 1000 N
    - Burn time: 5 s
    - Dry mass: 10 kg
    - Dry Inertia: (1.0, 1.0, 0.1)
    """
    thrust_curve = Function(lambda t: 1000 if t < 5 else 0, "Time (s)", "Thrust (N)")

    return SolidMotor(
        thrust_source=thrust_curve,
        burn_time=5,
        dry_mass=10.0,
        dry_inertia=(1.0, 1.0, 0.1),  # Ixx, Iyy, Izz
        grain_number=1,
        grain_density=1000,
        grain_outer_radius=0.05,
        grain_initial_inner_radius=0.02,
        grain_initial_height=0.5,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
        nozzle_radius=0.02,
        grain_separation=0.001,
        grains_center_of_mass_position=0.25,
        center_of_dry_mass_position=0.25,
    )


def test_cluster_initialization(base_motor):
    """
    Tests if the ClusterMotor initializes basic attributes correctly.
    """
    N = 3
    R = 0.5
    cluster = ClusterMotor(motor=base_motor, number=N, radius=R)

    assert cluster.number == N
    assert cluster.radius == R
    assert cluster.grain_outer_radius == base_motor.grain_outer_radius


def test_cluster_mass_and_thrust_scaling(base_motor):
    """
    Tests if scalar and derived properties are correctly multiplied by N and that functional properties preserve their Function behavior
    """
    N = 4
    R = 0.2
    cluster = ClusterMotor(motor=base_motor, number=N, radius=R)

    # 1. Check Thrust Scaling
    # Thrust at t=1 should be N * single_motor_thrust
    assert np.isclose(cluster.thrust(1), base_motor.thrust(1) * N)

    # 2. Check Dry Mass Scaling
    assert np.isclose(cluster.dry_mass, base_motor.dry_mass * N)

    # 3. Check Propellant Mass Scaling
    assert np.isclose(cluster.propellant_mass(0), base_motor.propellant_mass(0) * N)
    assert np.isclose(cluster.total_impulse, base_motor.total_impulse * N)
    assert np.isclose(cluster.average_thrust, base_motor.average_thrust * N)


def test_cluster_dry_inertia_steiner_theorem(base_motor):
    """
    Tests the implementation of the Parallel Axis Theorem (Huygens-Steiner)
    for the static (dry) mass of the cluster.

    Theoretical Formulas (exact for an evenly spaced ring of N >= 3 motors):
    I_zz_cluster = N * I_zz_local + N * m * R^2
    I_xx_cluster = N * I_xx_local + (N/2) * m * R^2
    """
    N = 3
    R = 1.0  # 1 meter radius for simpler checking
    cluster = ClusterMotor(motor=base_motor, number=N, radius=R)

    m_dry = base_motor.dry_mass
    Ixx_loc = base_motor.dry_I_11
    Izz_loc = base_motor.dry_I_33

    # Expected Izz (Longitudinal / Roll)
    expected_Izz = N * Izz_loc + N * m_dry * (R**2)

    # Expected Ixx/Iyy (Transverse / Pitch / Yaw)
    expected_I_trans = N * Ixx_loc + (N / 2) * m_dry * (R**2)

    assert np.isclose(cluster.dry_I_33, expected_Izz)
    assert np.isclose(cluster.dry_I_11, expected_I_trans)
    assert np.isclose(cluster.dry_I_22, expected_I_trans)


def test_cluster_propellant_inertia_dynamic(base_motor):
    """
    Tests if the Steiner theorem is correctly applied dynamically
    to the changing propellant mass over time.
    """
    N = 4
    R = 0.5
    cluster = ClusterMotor(motor=base_motor, number=N, radius=R)

    t = 0  # Check at t=0

    m_prop = base_motor.propellant_mass(t)
    Ixx_prop_loc = base_motor.propellant_I_11(t)
    Izz_prop_loc = base_motor.propellant_I_33(t)

    # Expected Dynamic Ixx
    # Ixx_term1 (Local rotation) + Ixx_term2 (Parallel axis offset)
    expected_Ixx = (Ixx_prop_loc * N) + (m_prop * 0.5 * N * R**2)

    # Expected Dynamic Izz
    expected_Izz = (Izz_prop_loc * N) + (m_prop * N * R**2)

    assert np.isclose(cluster.propellant_I_11(t), expected_Ixx)
    assert np.isclose(cluster.propellant_I_33(t), expected_Izz)


def test_cluster_invalid_inputs(base_motor):
    """Tests if the validation raises errors for bad inputs."""
    with pytest.raises(ValueError):
        ClusterMotor(motor=base_motor, number=1, radius=0.5)  # N < 2
    with pytest.raises(ValueError):
        ClusterMotor(motor=base_motor, number=2, radius=-1.0)  # Radius < 0
    with pytest.raises(TypeError):
        ClusterMotor(motor=base_motor, number="two", radius=0.5)  # N is string


def test_cluster_methods_and_setters(base_motor):
    """Touches the display methods and setters to ensure coverage."""
    cluster = ClusterMotor(motor=base_motor, number=2, radius=0.5)

    # 1. Touch the info method
    cluster.info()

    # 2. Touch the draw method (without showing the plot to avoid blocking tests)
    cluster.draw_cluster_layout(show=False)
    cluster.draw_cluster_layout(rocket_radius=0.1, show=False)

    # 3. Touch a few setters with values of the documented Function type
    cluster.propellant_mass = Function(50.0)
    assert cluster.propellant_mass(0) == 50.0

    cluster.propellant_I_11 = Function(2.0)
    assert cluster.propellant_I_11(0) == 2.0
