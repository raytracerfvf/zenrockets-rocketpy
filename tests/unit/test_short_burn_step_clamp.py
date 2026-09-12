"""Regression: an impulsive sub-second burn must not be skipped by the ODE solver.

With max_time_step=np.inf the LSODA initial step defaults to ~max_time*1e-3 (≈0.6 s
for max_time=600), which lands past burnout for short motors like the Cesaroni F240
(burn_out_time=0.328 s). Without the per-phase clamp in Flight.__simulate the
solver sees thrust=0 at both step endpoints and accepts a no-op step, leaving the
rocket on the pad.
"""

import numpy as np

from rocketpy import (
    CompositeMotor,
    Environment,
    Flight,
    MotorPlacement,
    Rocket,
    SolidMotor,
)


def _make_short_burn_motor():
    # Cesaroni F240-style: 0.328 s burn, ~67 N·s impulse, ~280 N peak.
    thrust_curve = [
        [0.000, 0.0],
        [0.010, 200.0],
        [0.050, 240.0],
        [0.150, 280.0],
        [0.250, 240.0],
        [0.300, 80.0],
        [0.328, 0.0],
    ]
    return SolidMotor(
        thrust_source=thrust_curve,
        burn_time=0.328,
        dry_mass=0.0615,
        dry_inertia=(9.29e-5, 9.29e-5, 8.86e-6),
        center_of_dry_mass_position=0.0665,
        nozzle_position=0.0,
        grain_number=2,
        grain_density=1820.0,
        grain_outer_radius=0.0104,
        grain_initial_inner_radius=0.00585,
        grain_initial_height=0.0359,
        grain_separation=0.0012,
        grains_center_of_mass_position=0.0635,
        nozzle_radius=0.0091,
        throat_radius=0.0034,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )


def _make_near_instant_burn_motor():
    # Placeholder with negligible impulse over a 1e-10 s burn; the clamp must
    # not impose burn_out/10 = 1e-11 s as max_step.
    return SolidMotor(
        thrust_source=1e-300,
        burn_time=1e-10,
        dry_mass=0.0615,
        dry_inertia=(9.29e-5, 9.29e-5, 8.86e-6),
        center_of_dry_mass_position=0.0665,
        nozzle_position=0.0,
        grain_number=2,
        grain_density=1e-300,
        grain_outer_radius=0.0104,
        grain_initial_inner_radius=0.00585,
        grain_initial_height=0.0359,
        grain_separation=0.0012,
        grains_center_of_mass_position=0.0635,
        nozzle_radius=0.0091,
        throat_radius=0.0034,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )


def _make_light_rocket(motor):
    rocket = Rocket(
        radius=0.025,
        mass=0.230,
        inertia=(0.013, 0.013, 5.0e-5),
        power_off_drag=lambda mach: 0.45,
        power_on_drag=lambda mach: 0.45,
        center_of_mass_without_motor=0.342,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=0.0)
    rocket.add_nose(length=0.1, kind="ogive", position=0.623)
    rocket.add_trapezoidal_fins(
        n=3,
        root_chord=0.05,
        tip_chord=0.025,
        span=0.04,
        position=0.05,
        sweep_length=0.01,
    )
    rocket.set_rail_buttons(
        upper_button_position=0.4,
        lower_button_position=0.1,
        angular_position=0,
    )
    return rocket


def test_short_burn_motor_lifts_off_with_default_max_time_step():
    motor = _make_short_burn_motor()
    rocket = _make_light_rocket(motor)
    env = Environment()

    flight = Flight(
        environment=env,
        rocket=rocket,
        rail_length=2.0,
        inclination=90,
        heading=0,
        terminate_on_apogee=True,
        max_time_step=np.inf,
    )

    assert flight.out_of_rail_velocity > 5, (
        f"rocket failed to clear rail (v={flight.out_of_rail_velocity:.3f} m/s); "
        "ODE solver likely stepped over the burn"
    )
    assert flight.apogee > env.elevation + 100, (
        f"apogee={flight.apogee:.1f} m too low for an F240-class motor; "
        "burn was probably skipped"
    )


def _make_slow_core_motor():
    # Gentle 2 s core burn that alone lifts the light rocket a little.
    return SolidMotor(
        thrust_source=[[0.0, 0.0], [0.05, 12.0], [1.9, 12.0], [2.0, 0.0]],
        burn_time=2.0,
        dry_mass=0.06,
        dry_inertia=(9.0e-5, 9.0e-5, 9.0e-6),
        center_of_dry_mass_position=0.06,
        nozzle_position=0.0,
        grain_number=1,
        grain_density=1820.0,
        grain_outer_radius=0.0104,
        grain_initial_inner_radius=0.004,
        grain_initial_height=0.05,
        grain_separation=0.0,
        grains_center_of_mass_position=0.06,
        nozzle_radius=0.006,
        throat_radius=0.003,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )


def _fly(motor):
    rocket = _make_light_rocket(motor)
    flight = Flight(
        environment=Environment(),
        rocket=rocket,
        rail_length=2.0,
        inclination=90,
        heading=0,
        terminate_on_apogee=True,
        max_time_step=np.inf,
    )
    return flight.apogee


def test_late_short_pulse_in_a_composite_is_not_skipped():
    core = _make_slow_core_motor()
    pulse = _make_short_burn_motor()
    core_only = CompositeMotor([MotorPlacement(core, 0.0)])
    with_pulse = CompositeMotor(
        [MotorPlacement(core, 0.0), MotorPlacement(pulse, 0.0, ignition_delay=3.0)]
    )

    assert with_pulse.min_burn_duration == pulse.burn_duration
    assert _fly(with_pulse) > _fly(core_only) + 100, (
        "the 0.33 s pulse igniting at t=3 s added no altitude; "
        "the solver stepped over it"
    )


def test_near_instant_burn_does_not_stall_solver():
    """A degenerate near-zero burn must not clamp max_step into a stall.

    Regression for the burn-phase clamp: burn_out/10 = 1e-11 s for this motor.
    Before the MIN_BURN_CLAMP_STEP guard the solver was pinned to that step and
    crawled toward max_time (~1e11 steps), hanging the simulation indefinitely.
    """
    motor = _make_near_instant_burn_motor()
    rocket = _make_light_rocket(motor)

    # Positive impulse (clamp's first gate) but a degenerate burn window.
    assert motor.total_impulse > 0

    flight = Flight(
        environment=Environment(),
        rocket=rocket,
        rail_length=2.0,
        inclination=90,
        heading=0,
        max_time=2.0,
        max_time_step=1e-2,
    )

    # With the clamp skipped, max_time_step=1e-2 over max_time=2 s keeps the
    # solution bounded (~200 points). A regressed clamp would explode this.
    assert len(flight.solution) < 5000


def test_placeholder_does_not_disable_composite_burn_clamp():
    pulse = _make_short_burn_motor()
    placeholder = _make_near_instant_burn_motor()
    # No ballast from the placeholder, so only step selection differs.
    placeholder.dry_mass = 0
    placeholder.dry_I_11 = placeholder.dry_I_22 = placeholder.dry_I_33 = 0
    composite = CompositeMotor(
        [MotorPlacement(pulse, 0), MotorPlacement(placeholder, 0)]
    )
    assert composite.min_burn_duration == pulse.burn_duration
    assert np.isclose(_fly(composite), _fly(pulse), rtol=0.01)
