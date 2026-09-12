# pylint: disable=invalid-name
"""A motor assembled from several motors placed around and along the rocket
axis, exposed to ``Rocket`` and ``Flight`` as a single ``Motor``."""

from dataclasses import dataclass
from functools import cached_property
from math import sqrt

import numpy as np

from rocketpy.mathutils.function import Function, funcify_method
from rocketpy.motors.motor import MIN_BURN_CLAMP_STEP, Motor
from rocketpy.tools import parallel_axis_theorem_from_com

# Zero guard points this far outside each burn window keep linear
# interpolation from bleeding thrust into a gap.
_GUARD_DT = 1e-6
# Uniform samples per burn window so mass and inertia interpolate well
# between sparse thrust breakpoints.
_SAMPLES_PER_BURN = 200


@dataclass(frozen=True)
class MotorPlacement:
    """One physical motor inside a ``CompositeMotor``.

    Attributes
    ----------
    motor : Motor
        The motor. Its coordinate system must be
        ``nozzle_to_combustion_chamber``.
    position : float
        Position of the motor's own origin along the composite axis, in meters
        from the composite origin, positive toward the combustion chamber.
    lateral : tuple[float, float]
        Lateral (x, y) offset of the motor axis from the composite axis, in
        meters, in the rocket body frame. ``(0, 0)`` is on the axis.
    ignition_delay : float
        Seconds after composite ignition at which this motor ignites.
    """

    motor: Motor
    position: float
    lateral: tuple[float, float] = (0.0, 0.0)
    ignition_delay: float = 0.0


def _stored(name):
    """A settable property backed by ``self._<name>``; satisfies the abstract
    ``Motor`` interface while keeping the attribute assignable."""
    attr = f"_{name}"
    return property(
        lambda self: getattr(self, attr),
        lambda self, value: setattr(self, attr, value),
        doc=f"{name} of the composite as a Function of time.",
    )


class CompositeMotor(Motor):
    """Several motors aggregated into one.

    Thrust, mass, mass flow and propellant properties are the sums of the
    placed motors, each shifted by its ignition delay. Inertia applies the
    parallel axis theorem per motor with its lateral and axial offsets. The
    tensor and reported centers of mass use an on-axis reference. Laterally
    asymmetric layouts have an off-axis true center of mass and thrust moments
    that this aggregation does not model. Time-dependent mass properties are
    precomputed on one shared time grid as array-sourced ``Function`` objects.

    Attributes
    ----------
    placements : list[MotorPlacement]
        The placed motors.
    min_burn_duration : float
        Shortest non-degenerate positive-impulse burn, used by ``Flight`` to
        bound the solver step; zero if none qualifies.
    dry_lateral_mass_moment : tuple[float, float]
        Dry mass lateral moments (Σ m x, Σ m y) in kg m, body frame.
    propellant_lateral_mass_moment : tuple[Function, Function]
        Propellant lateral moments (Σ m x, Σ m y) in kg m as functions of time.
    """

    def __init__(self, placements):
        """
        Parameters
        ----------
        placements : sequence of MotorPlacement
            At least one placement. All motors must use the
            ``nozzle_to_combustion_chamber`` coordinate system.
        """
        placements = list(placements)
        if not placements:
            raise ValueError("CompositeMotor needs at least one MotorPlacement")
        for placement in placements:
            if placement.motor._csys != 1:
                raise ValueError(
                    "CompositeMotor sub-motors must use the "
                    "'nozzle_to_combustion_chamber' coordinate system"
                )
        self.placements = placements
        self._grid = self._time_grid()

        thrust = self._sum(lambda p: self._sample(p, p.motor.thrust, outside=0.0))
        dry_masses = np.array([p.motor.dry_mass for p in placements])
        dry_cms = np.array(
            [p.position + p.motor.center_of_dry_mass_position for p in placements]
        )
        dry_mass = float(dry_masses.sum())
        center_of_dry_mass = (
            float((dry_masses * dry_cms).sum() / dry_mass) if dry_mass else 0.0
        )
        lateral = np.array([p.lateral for p in placements])
        self.dry_lateral_mass_moment = tuple(
            float(value) for value in (dry_masses[:, None] * lateral).sum(axis=0)
        )
        dry_inertia = self._inertia_about_axis_point(
            masses=dry_masses[:, None],
            axial=dry_cms[:, None],
            reference=center_of_dry_mass,
            own=[
                np.array(
                    [
                        getattr(p.motor, f"dry_I_{component}")
                        for component in ("11", "22", "33", "12", "13", "23")
                    ]
                )[:, None]
                for p in placements
            ],
        )

        super().__init__(
            thrust_source=np.column_stack([self._grid, thrust]),
            dry_inertia=tuple(float(component[0]) for component in dry_inertia),
            nozzle_radius=sqrt(sum(p.motor.nozzle_radius**2 for p in placements)),
            center_of_dry_mass_position=center_of_dry_mass,
            dry_mass=dry_mass,
            nozzle_position=min(
                p.position + p.motor.nozzle_position for p in placements
            ),
            burn_time=(
                0.0,
                max(p.ignition_delay + p.motor.burn_out_time for p in placements),
            ),
            interpolation_method="linear",
        )
        self.min_burn_duration = min(
            (
                p.motor.burn_duration
                for p in placements
                if p.motor.total_impulse > 0
                and p.motor.burn_duration / 10 >= MIN_BURN_CLAMP_STEP
            ),
            default=0.0,
        )

        self._propellant_initial_mass = sum(
            p.motor.propellant_initial_mass for p in placements
        )
        propellant_masses = np.array(
            [self._sample(p, p.motor.propellant_mass) for p in placements]
        )
        self.propellant_lateral_mass_moment = (
            self._function(
                (propellant_masses * lateral[:, 0, None]).sum(axis=0),
                "Propellant lateral mass moment x (kg m)",
            ),
            self._function(
                (propellant_masses * lateral[:, 1, None]).sum(axis=0),
                "Propellant lateral mass moment y (kg m)",
            ),
        )
        propellant_cms = np.array(
            [
                p.position + self._sample(p, p.motor.center_of_propellant_mass)
                for p in placements
            ]
        )
        total_propellant = propellant_masses.sum(axis=0)
        center_of_propellant = np.divide(
            (propellant_masses * propellant_cms).sum(axis=0),
            total_propellant,
            out=np.full_like(total_propellant, center_of_dry_mass),
            where=total_propellant > 0,
        )
        propellant_inertia = self._inertia_about_axis_point(
            masses=propellant_masses,
            axial=propellant_cms,
            reference=center_of_propellant,
            own=[
                np.array(
                    [
                        self._sample(p, getattr(p.motor, f"propellant_I_{component}"))
                        for component in ("11", "22", "33", "12", "13", "23")
                    ]
                )
                for p in placements
            ],
        )

        self.propellant_mass = self._function(total_propellant, "Propellant mass (kg)")
        self.total_mass_flow_rate = self._function(
            self._sum(
                lambda p: self._sample(p, p.motor.total_mass_flow_rate, outside=0.0)
            ),
            "Mass flow rate (kg/s)",
            extrapolation="zero",
        )
        self.mass_flow_rate = self.total_mass_flow_rate
        self._center_of_propellant_mass = self._function(
            center_of_propellant, "Propellant center of mass (m)"
        )
        for component, values in zip(
            ("11", "22", "33", "12", "13", "23"), propellant_inertia
        ):
            setattr(
                self,
                f"_propellant_I_{component}",
                self._function(values, f"Inertia I_{component} (kg m²)"),
            )

    def _time_grid(self):
        """Union of every shifted thrust breakpoint and a uniform sampling of
        each burn window, plus zero guard points just outside each window."""
        knots = [np.array([0.0])]
        for p in self.placements:
            start, end = p.motor.burn_time
            knots.append(p.motor.thrust.x_array + p.ignition_delay)
            knots.append(
                np.linspace(start, end, _SAMPLES_PER_BURN + 1) + p.ignition_delay
            )
            knots.append(
                np.array(
                    [
                        start + p.ignition_delay - _GUARD_DT,
                        end + p.ignition_delay + _GUARD_DT,
                    ]
                )
            )
        grid = np.unique(np.concatenate(knots))
        return grid[grid >= 0]

    def _sample(self, placement, function, outside=None):
        """Evaluate a sub-motor Function on the grid, shifted by the placement's
        ignition delay. Outside the motor's burn window the value is ``outside``
        when given (thrust, mass flow), else the window edge value (mass, CG,
        inertia are constant before ignition and after burnout)."""
        start, end = placement.motor.burn_time
        local_time = self._grid - placement.ignition_delay
        if outside is None:
            return np.array(
                [function.get_value_opt(min(max(t, start), end)) for t in local_time]
            )
        return np.array(
            [
                function.get_value_opt(t) if start <= t <= end else outside
                for t in local_time
            ]
        )

    def _sum(self, sampler):
        return sum(sampler(p) for p in self.placements)

    def _function(self, values, label, extrapolation="constant"):
        return Function(
            np.column_stack([self._grid, values]),
            "Time (s)",
            label,
            "linear",
            extrapolation,
        )

    def _inertia_about_axis_point(self, masses, axial, reference, own):
        """Sum the placed motors' inertia tensors about the point on the
        composite axis at ``reference``, applying the parallel axis theorem
        with each motor's lateral (x, y) and axial offsets. Products of inertia
        follow RocketPy's sign convention (``I_12 = -Σ m x y``). Arrays are
        (placements, samples); ``own`` holds each motor's six components."""
        lateral = np.array([p.lateral for p in self.placements])
        x = lateral[:, :1]
        y = lateral[:, 1:]
        dz = axial - reference
        m = masses
        return (
            sum(o[0] for o in own) + (m * (y**2 + dz**2)).sum(axis=0),
            sum(o[1] for o in own) + (m * (x**2 + dz**2)).sum(axis=0),
            sum(o[2] for o in own) + (m * (x**2 + y**2)).sum(axis=0),
            sum(o[3] for o in own) - (m * x * y).sum(axis=0),
            sum(o[4] for o in own) - (m * x * dz).sum(axis=0),
            sum(o[5] for o in own) - (m * y * dz).sum(axis=0),
        )

    def pressure_thrust(self, pressure, t=None):
        """Sum pressure corrections for motors burning at composite time ``t``.

        If ``t`` is omitted, treat all motors as active. Motors without a
        reference pressure contribute no correction.
        """
        correction = 0.0
        for p in self.placements:
            motor = p.motor
            if motor.reference_pressure is None:
                continue
            if t is None or (
                motor.burn_start_time <= t - p.ignition_delay <= motor.burn_out_time
            ):
                correction += (motor.reference_pressure - pressure) * motor.nozzle_area
        return correction

    @cached_property
    def vacuum_thrust(self):
        """Thrust plus each active motor's reference-pressure contribution."""
        correction = np.zeros_like(self._grid)
        for p in self.placements:
            if p.motor.reference_pressure is not None:
                contribution = Function(
                    p.motor.reference_pressure * p.motor.nozzle_area
                )
                correction += self._sample(p, contribution, outside=0.0)
        return self.thrust + self._function(
            correction, "Vacuum pressure correction (N)", extrapolation="zero"
        )

    @funcify_method("Time (s)", "Exhaust velocity (m/s)")
    def exhaust_velocity(self):
        """Constant average exhaust velocity, total impulse over propellant
        mass, as ``SolidMotor`` assumes."""
        return Function(
            self.total_impulse / self.propellant_initial_mass
        ).set_discrete_based_on_model(self.thrust)

    @property
    def propellant_initial_mass(self):
        """Initial propellant mass of all placed motors, kg."""
        return self._propellant_initial_mass

    center_of_propellant_mass = _stored("center_of_propellant_mass")
    propellant_I_11 = _stored("propellant_I_11")
    propellant_I_22 = _stored("propellant_I_22")
    propellant_I_33 = _stored("propellant_I_33")
    propellant_I_12 = _stored("propellant_I_12")
    propellant_I_13 = _stored("propellant_I_13")
    propellant_I_23 = _stored("propellant_I_23")

    @funcify_method("Time (s)", "Inertia I_22 (kg m²)")
    def I_22(self):
        """Inertia about e_2 at the instantaneous center of mass. The base
        class assumes I_22 = I_11, which an asymmetric layout breaks."""
        prop_to_cm = self.center_of_propellant_mass - self.center_of_mass
        dry_to_cm = self.center_of_dry_mass_position - self.center_of_mass
        return parallel_axis_theorem_from_com(
            self.propellant_I_22, self.propellant_mass, prop_to_cm
        ) + parallel_axis_theorem_from_com(self.dry_I_22, self.dry_mass, dry_to_cm)

    @funcify_method("Time (s)", "Inertia I_13 (kg m²)")
    def I_13(self):
        """Cross-inertia about the on-axis instantaneous center of mass."""
        # I_13 = -Σ m x (z - z_ref), so moving z_ref to z_cm adds (z_cm - z_ref) Σ m x
        return (
            self.dry_I_13
            + self.propellant_I_13
            + (self.center_of_mass - self.center_of_dry_mass_position)
            * self.dry_lateral_mass_moment[0]
            + (self.center_of_mass - self.center_of_propellant_mass)
            * self.propellant_lateral_mass_moment[0]
        )

    @funcify_method("Time (s)", "Inertia I_23 (kg m²)")
    def I_23(self):
        """Cross-inertia about the on-axis instantaneous center of mass."""
        return (
            self.dry_I_23
            + self.propellant_I_23
            + (self.center_of_mass - self.center_of_dry_mass_position)
            * self.dry_lateral_mass_moment[1]
            + (self.center_of_mass - self.center_of_propellant_mass)
            * self.propellant_lateral_mass_moment[1]
        )

    def info(self, *, filename=None):
        print(f"Composite of {len(self.placements)} motors:")
        for index, p in enumerate(self.placements, start=1):
            x, y = p.lateral
            print(
                f" - #{index} {type(p.motor).__name__} at z={p.position:.3f} m, "
                f"(x, y)=({x:.3f}, {y:.3f}) m, ignition +{p.ignition_delay:.2f} s"
            )
        return super().info(filename=filename)
