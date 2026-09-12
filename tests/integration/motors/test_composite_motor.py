# pylint: disable=invalid-name
import numpy as np
import pytest

from rocketpy import (
    CompositeMotor,
    Environment,
    Flight,
    Function,
    MotorPlacement,
    Rocket,
    SolidMotor,
)
from rocketpy.motors.cluster_motor import ClusterMotor


def _solid(
    thrust, burn_time, dry_mass, cdm=0.25, grain_cm=None, reference_pressure=None
):
    return SolidMotor(
        thrust_source=Function(
            lambda t: thrust if t < burn_time else 0, "Time (s)", "Thrust (N)"
        ),
        burn_time=burn_time,
        dry_mass=dry_mass,
        dry_inertia=(1.0, 1.0, 0.1),
        grain_number=1,
        grain_density=1000,
        grain_outer_radius=0.05,
        grain_initial_inner_radius=0.02,
        grain_initial_height=0.5,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
        nozzle_radius=0.02,
        grain_separation=0.001,
        grains_center_of_mass_position=cdm if grain_cm is None else grain_cm,
        center_of_dry_mass_position=cdm,
        reference_pressure=reference_pressure,
    )


@pytest.fixture
def core():
    """1000 N for 5 s, 10 kg dry."""
    return _solid(1000, 5, 10.0)


@pytest.fixture
def small():
    """200 N for 2 s, 2 kg dry."""
    return _solid(200, 2, 2.0)


def test_single_placement_matches_the_motor(core):
    composite = CompositeMotor([MotorPlacement(core, 0.0)])

    for t in (0, 1, 2.5, 4.9):
        assert np.isclose(composite.thrust(t), core.thrust(t))
        assert np.isclose(composite.propellant_mass(t), core.propellant_mass(t))
        assert np.isclose(composite.total_mass(t), core.total_mass(t))
        assert np.isclose(
            composite.center_of_propellant_mass(t), core.center_of_propellant_mass(t)
        )
        # Interpolated on the composite grid, so agreement is to grid resolution
        assert np.isclose(
            composite.propellant_I_11(t), core.propellant_I_11(t), rtol=1e-4
        )
        assert np.isclose(
            composite.propellant_I_33(t), core.propellant_I_33(t), rtol=1e-4
        )
    assert np.isclose(composite.total_impulse, core.total_impulse)
    assert composite.burn_time == core.burn_time
    assert composite.dry_mass == core.dry_mass
    assert composite.center_of_dry_mass_position == core.center_of_dry_mass_position
    assert composite.dry_I_11 == core.dry_I_11
    assert composite.dry_I_33 == core.dry_I_33
    assert composite.nozzle_radius == core.nozzle_radius
    assert composite.min_burn_duration == core.burn_duration


def test_ring_matches_cluster_motor(core):
    N, R = 3, 0.4
    cluster = ClusterMotor(core, N, R)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    composite = CompositeMotor(
        [MotorPlacement(core, 0.0, (R * np.cos(a), R * np.sin(a))) for a in angles]
    )

    assert np.isclose(composite.thrust(1), cluster.thrust(1))
    assert np.isclose(composite.dry_I_11, cluster.dry_I_11)
    assert np.isclose(composite.dry_I_33, cluster.dry_I_33)
    assert np.isclose(composite.propellant_I_11(1), cluster.propellant_I_11(1))
    assert np.isclose(composite.propellant_I_33(1), cluster.propellant_I_33(1))
    # Closed form for an evenly spaced ring
    assert np.isclose(composite.dry_I_33, N * core.dry_I_33 + N * core.dry_mass * R**2)
    assert np.isclose(
        composite.dry_I_11, N * core.dry_I_11 + N / 2 * core.dry_mass * R**2
    )


def test_two_motors_on_one_line_are_anisotropic(core):
    R = 0.5
    composite = CompositeMotor(
        [MotorPlacement(core, 0.0, (R, 0.0)), MotorPlacement(core, 0.0, (-R, 0.0))]
    )
    m = core.dry_mass
    # Both motors sit on the x axis: nothing to shift about e_1, m r² each about e_2
    assert np.isclose(composite.dry_I_11, 2 * core.dry_I_11)
    assert np.isclose(composite.dry_I_22, 2 * core.dry_I_22 + 2 * m * R**2)
    assert np.isclose(composite.dry_I_33, 2 * core.dry_I_33 + 2 * m * R**2)
    assert composite.dry_I_12 == 0
    # The axial Steiner shift is the same about both axes, so the full tensor
    # keeps exactly the dry and propellant anisotropy
    assert np.isclose(
        composite.I_22(0) - composite.I_11(0),
        (composite.dry_I_22 - composite.dry_I_11)
        + (composite.propellant_I_22(0) - composite.propellant_I_11(0)),
    )


def test_single_off_axis_motor_has_products_of_inertia(core):
    x, y = 0.3, 0.2
    composite = CompositeMotor([MotorPlacement(core, 0.0, (x, y))])
    m = core.dry_mass
    assert np.isclose(composite.dry_I_11, core.dry_I_11 + m * y**2)
    assert np.isclose(composite.dry_I_22, core.dry_I_22 + m * x**2)
    assert np.isclose(composite.dry_I_33, core.dry_I_33 + m * (x**2 + y**2))
    assert np.isclose(composite.dry_I_12, -m * x * y)
    # Dry CM is on the axis at the motor's own axial CM, so no axial products
    assert composite.dry_I_13 == 0
    assert composite.dry_I_23 == 0
    mp = core.propellant_mass(0)
    assert np.isclose(composite.propellant_I_12(0), -mp * x * y)
    assert composite.I_22(0) > composite.I_11(0)


def test_delayed_second_motor(core, small):
    delay = 6.0
    composite = CompositeMotor(
        [
            MotorPlacement(core, 0.0),
            MotorPlacement(small, 0.1, (0.2, 0.0), ignition_delay=delay),
        ]
    )

    assert composite.burn_time == (0.0, delay + small.burn_out_time)
    assert composite.min_burn_duration == small.burn_duration
    assert np.isclose(composite.total_impulse, core.total_impulse + small.total_impulse)
    assert np.isclose(
        composite.propellant_initial_mass,
        core.propellant_initial_mass + small.propellant_initial_mass,
    )
    # Only the core burns first, nothing burns in the gap, then only the small motor
    assert np.isclose(composite.thrust(1), 1000)
    assert np.isclose(composite.thrust(5.5), 0)
    assert np.isclose(composite.thrust(7), 200)
    assert np.isclose(composite.total_mass_flow_rate(5.5), 0)
    # Propellant of the delayed motor stays full until it ignites
    assert np.isclose(composite.propellant_mass(5.5), small.propellant_initial_mass)
    assert np.isclose(
        composite.propellant_mass(delay + 1),
        small.propellant_mass(1),
    )
    assert np.isclose(composite.propellant_mass(0), composite.propellant_initial_mass)
    assert composite.propellant_mass(delay + small.burn_out_time) < 1e-6
    times = np.linspace(0, composite.burn_out_time, 200)
    masses = [composite.total_mass(t) for t in times]
    assert all(a >= b - 1e-9 for a, b in zip(masses, masses[1:]))


def test_axial_offsets_move_the_centers_of_mass(core):
    composite = CompositeMotor([MotorPlacement(core, 0.0), MotorPlacement(core, 1.0)])
    assert np.isclose(
        composite.center_of_dry_mass_position, core.center_of_dry_mass_position + 0.5
    )
    assert np.isclose(
        composite.center_of_propellant_mass(0), core.center_of_propellant_mass(0) + 0.5
    )
    assert composite.nozzle_position == core.nozzle_position
    # Two equal masses 1 m apart about their midpoint add 2 · m · 0.5² transverse
    assert np.isclose(composite.dry_I_11, 2 * core.dry_I_11 + 2 * core.dry_mass * 0.25)
    assert np.isclose(composite.dry_I_33, 2 * core.dry_I_33)


def test_invalid_inputs(core):
    with pytest.raises(ValueError):
        CompositeMotor([])
    flipped = _solid(1000, 5, 10.0)
    flipped._csys = -1
    with pytest.raises(ValueError):
        CompositeMotor([MotorPlacement(flipped, 0.0)])


def test_setters_and_info(core, capsys):
    composite = CompositeMotor([MotorPlacement(core, 0.0)])
    composite.propellant_mass = Function(50.0)
    assert composite.propellant_mass(0) == 50.0
    composite.propellant_I_11 = Function(2.0)
    assert composite.propellant_I_11(0) == 2.0
    composite.info()
    assert "Composite of 1 motors" in capsys.readouterr().out


def test_composite_flies(core, small):
    composite = CompositeMotor(
        [
            MotorPlacement(core, 0.0),
            MotorPlacement(small, 0.0, (0.15, 0.0), ignition_delay=5.5),
        ]
    )
    rocket = Rocket(
        radius=0.1,
        mass=20.0,
        inertia=(6.0, 6.0, 0.1),
        power_off_drag=lambda mach: 0.5,
        power_on_drag=lambda mach: 0.5,
        center_of_mass_without_motor=1.5,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(composite, position=0)
    rocket.add_nose(length=0.3, kind="ogive", position=2.5)
    rocket.add_trapezoidal_fins(
        n=4, root_chord=0.2, tip_chord=0.1, span=0.15, position=0.1
    )
    flight = Flight(
        environment=Environment(),
        rocket=rocket,
        rail_length=5.0,
        inclination=90,
        heading=0,
        terminate_on_apogee=True,
    )
    assert flight.apogee_time > composite.burn_out_time
    assert flight.apogee > flight.env.elevation + 500


def test_single_motor_pressure_correction():
    motor = _solid(1000, 5, 10, reference_pressure=101325)
    composite = CompositeMotor([MotorPlacement(motor, 0)])
    assert composite.pressure_thrust(80000) == pytest.approx(
        motor.pressure_thrust(80000)
    )
    assert composite.pressure_thrust(80000, t=1) == pytest.approx(
        (101325 - 80000) * motor.nozzle_area
    )
    assert motor.pressure_thrust(80000, t=10) == motor.pressure_thrust(80000)


@pytest.mark.parametrize("delay", [3.0, 6.0])
def test_pressure_correction_uses_active_burn_windows(delay):
    first = _solid(1000, 5, 10, reference_pressure=101325)
    second = _solid(200, 2, 2, reference_pressure=95000)
    unknown = _solid(200, 2, 2)
    composite = CompositeMotor(
        [
            MotorPlacement(first, 0),
            MotorPlacement(second, 0, ignition_delay=delay),
            MotorPlacement(unknown, 0),
        ]
    )
    pressure = 80000
    first_correction = (101325 - pressure) * first.nozzle_area
    second_correction = (95000 - pressure) * second.nozzle_area
    assert composite.pressure_thrust(pressure) == pytest.approx(
        first_correction + second_correction
    )
    for t in (-1, 0, 1, 3, 4, 5, 5.5, 6, 7, 8, 9):
        expected = (first_correction if 0 <= t <= 5 else 0) + (
            second_correction if delay <= t <= delay + 2 else 0
        )
        assert composite.pressure_thrust(pressure, t=t) == pytest.approx(expected)


@pytest.mark.parametrize("delay", [3.0, 6.0])
def test_vacuum_thrust_uses_active_burn_windows(delay):
    first = _solid(1000, 5, 10, reference_pressure=101325)
    second = _solid(200, 2, 2, reference_pressure=95000)
    composite = CompositeMotor(
        [MotorPlacement(first, 0), MotorPlacement(second, 0, ignition_delay=delay)]
    )
    for t in (-1, 0, 1, 4, 5, 5.5, 6, 7, 8, 9):
        correction = (101325 * first.nozzle_area if 0 <= t <= 5 else 0) + (
            95000 * second.nozzle_area if delay <= t <= delay + 2 else 0
        )
        assert composite.vacuum_thrust(t) == pytest.approx(
            composite.thrust(t) + correction
        )


def _rocket_with_motor(motor, orientation="tail_to_nose"):
    sign = 1 if orientation == "tail_to_nose" else -1
    rocket = Rocket(
        radius=0.1,
        mass=20,
        inertia=(6, 6, 0.1),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=sign * 1.5,
        coordinate_system_orientation=orientation,
    )
    rocket.add_motor(motor, position=sign * 0.4)
    return rocket


def _direct_cross_inertia(placements, t, reference, lateral_axis, dry_only=False):
    # Fixture motors have no own cross-inertia; sum dry and propellant terms
    # directly about the on-axis reference.
    result = 0.0
    for p in placements:
        motor = p.motor
        x = p.lateral[lateral_axis]
        dry_z = p.position + motor.center_of_dry_mass_position
        result -= motor.dry_mass * x * (dry_z - reference)
        if not dry_only:
            prop_z = p.position + motor.center_of_propellant_mass(t)
            result -= motor.propellant_mass(t) * x * (prop_z - reference)
    return result


@pytest.fixture
def asymmetric_composite():
    first = _solid(1000, 5, 10, cdm=0.2, grain_cm=0.5)
    second = _solid(200, 2, 2, cdm=0.1, grain_cm=0.4)
    return CompositeMotor(
        [
            MotorPlacement(first, 0, (0.3, -0.2)),
            MotorPlacement(second, 1.0, (-0.1, 0.4)),
        ]
    )


@pytest.mark.parametrize("component, lateral_axis", [("13", 0), ("23", 1)])
def test_cross_inertia_about_composite_cm(
    asymmetric_composite, component, lateral_axis
):
    composite = asymmetric_composite
    for t in (0, 1, 3, 5, 6):
        expected = _direct_cross_inertia(
            composite.placements, t, composite.center_of_mass(t), lateral_axis
        )
        assert getattr(composite, f"I_{component}")(t) == pytest.approx(
            expected, abs=1e-8
        )


@pytest.mark.parametrize("orientation", ["tail_to_nose", "nose_to_tail"])
@pytest.mark.parametrize("component, lateral_axis", [("13", 0), ("23", 1)])
def test_cross_inertia_about_rocket_cdm(
    asymmetric_composite, orientation, component, lateral_axis
):
    composite = asymmetric_composite
    rocket = _rocket_with_motor(composite, orientation)
    # Express the rocket CDM in the physical motor frame, positive noseward.
    reference = (
        rocket.center_of_dry_mass_position - rocket.motor_position
    ) * rocket._csys
    dry_expected = _direct_cross_inertia(
        composite.placements, 0, reference, lateral_axis, dry_only=True
    )
    assert getattr(rocket, f"dry_I_{component}") == pytest.approx(dry_expected)
    for t in (0, 1, 3, 5, 6):
        expected = _direct_cross_inertia(
            composite.placements, t, reference, lateral_axis
        )
        assert getattr(rocket, f"I_{component}")(t) == pytest.approx(expected, abs=1e-8)


def test_symmetric_ring_cross_inertias_remain_zero():
    motor = _solid(1000, 5, 10, cdm=0.2, grain_cm=0.5)
    composite = ClusterMotor(motor, 4, 0.3)
    rocket = _rocket_with_motor(composite)
    for component in ("13", "23"):
        assert getattr(rocket, f"dry_I_{component}") == pytest.approx(0, abs=1e-12)
        for t in (0, 1, 5, 6):
            assert getattr(composite, f"I_{component}")(t) == pytest.approx(
                0, abs=1e-12
            )
            assert getattr(rocket, f"I_{component}")(t) == pytest.approx(0, abs=1e-12)


def test_zero_dry_mass_remains_finite():
    motor = _solid(1000, 5, 0, cdm=0, grain_cm=0)
    composite = CompositeMotor([MotorPlacement(motor, 0)])
    assert composite.center_of_dry_mass_position == motor.center_of_dry_mass_position
    raw_rocket = _rocket_with_motor(motor)
    rocket = _rocket_with_motor(composite)
    for component in ("11", "22", "33", "12", "13", "23"):
        assert getattr(composite, f"dry_I_{component}") == pytest.approx(
            getattr(motor, f"dry_I_{component}")
        )
        assert getattr(rocket, f"dry_I_{component}") == pytest.approx(
            getattr(raw_rocket, f"dry_I_{component}")
        )
        for t in (0, 1, 5, 6):
            assert np.isfinite(getattr(rocket, f"I_{component}")(t))
            assert getattr(rocket, f"I_{component}")(t) == pytest.approx(
                getattr(raw_rocket, f"I_{component}")(t), rel=1e-4, abs=1e-8
            )
    for t in (0, 1, 5, 6):
        assert composite.center_of_propellant_mass(
            t
        ) == motor.center_of_propellant_mass(t)
        assert np.isfinite(rocket.center_of_mass(t))
        assert rocket.center_of_mass(t) == pytest.approx(raw_rocket.center_of_mass(t))
