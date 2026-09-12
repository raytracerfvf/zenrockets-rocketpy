# pylint: disable=invalid-name
import matplotlib.pyplot as plt
import numpy as np

from rocketpy.motors.composite_motor import CompositeMotor, MotorPlacement


class ClusterMotor(CompositeMotor):
    """
    A cluster of N identical motors evenly spaced on a ring around the rocket
    axis, all igniting together. A ``CompositeMotor`` with a ring layout.

    Attributes
    ----------
    motor : SolidMotor
        The single motor instance used in the cluster.
    number : int
        The number of motors in the cluster.
    radius : float
        The radial distance from the rocket's central axis to the center of each motor.
    """

    def __init__(self, motor, number, radius):
        """
        Initialize the ClusterMotor.

        Parameters
        ----------
        motor : SolidMotor
            The base motor to be clustered.
        number : int
            Number of motors. Must be >= 2.
        radius : float
            Distance from center of rocket to center of motor (m).
        """
        if not isinstance(number, int):
            raise TypeError(f"number must be an int, got {type(number).__name__}")
        if number < 2:
            raise ValueError("number must be >= 2 for a ClusterMotor")
        if not isinstance(radius, (int, float)):
            raise TypeError(
                f"radius must be a real number, got {type(radius).__name__}"
            )
        if radius < 0:
            raise ValueError("radius must be non-negative")

        self.motor = motor
        self.number = number
        self.radius = float(radius)

        angles = np.linspace(0, 2 * np.pi, number, endpoint=False)
        super().__init__(
            MotorPlacement(
                motor,
                position=0.0,
                lateral=(self.radius * np.cos(angle), self.radius * np.sin(angle)),
            )
            for angle in angles
        )
        self._setup_grain_properties()

    def _setup_grain_properties(self):
        """Copies the grain properties from the base motor."""
        self.throat_radius = self.motor.throat_radius
        self.grain_number = self.motor.grain_number
        self.grain_density = self.motor.grain_density
        self.grain_outer_radius = self.motor.grain_outer_radius
        self.grain_initial_inner_radius = self.motor.grain_initial_inner_radius
        self.grain_initial_height = self.motor.grain_initial_height
        self.grains_center_of_mass_position = self.motor.grains_center_of_mass_position

    def info(self, *, filename=None):
        print("Cluster Configuration:")
        print(f" - Motors: {self.number} x {type(self.motor).__name__}")
        print(f" - Radial Distance: {self.radius} m")
        return self.motor.info(filename=filename)

    def draw_cluster_layout(self, rocket_radius=None, show=True):
        """Draw the geometric layout of the clustered motors."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(0, 0, "k+", markersize=10, label="Central axis")
        if rocket_radius:
            rocket_tube = plt.Circle(
                (0, 0),
                rocket_radius,
                color="black",
                fill=False,
                linestyle="--",
                linewidth=2,
                label="Rocket",
            )
            ax.add_patch(rocket_tube)
            limit = rocket_radius * 1.2
        else:
            limit = self.radius * 2
        self._draw_engines(ax)
        ax.set_aspect("equal", "box")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xlabel("Position X (m)")
        ax.set_ylabel("Position Y (m)")
        ax.set_title(f"Cluster Configuration : {self.number} engines")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")
        if show:
            plt.show()
        return fig, ax

    def _draw_engines(self, ax):
        """Draws the individual engines of the cluster."""
        for i, placement in enumerate(self.placements):
            x, y = placement.lateral
            motor_circle = plt.Circle(
                (x, y),
                self.grain_outer_radius,
                color="red",
                alpha=0.5,
                label="Engine" if i == 0 else "",
            )
            ax.add_patch(motor_circle)
            ax.text(
                x,
                y,
                str(i + 1),
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
            )
