import numpy as np
from src.pose_estimation.nonlinear_system import NonlinearSystem


class DifferentialDriveSystem(NonlinearSystem):
    """
    Nonlinear system model for a differential-drive robot (DDR).

    Defines the motion and measurement models along with their Jacobians
    overriding the NonlinearSystem super class.

    Attributes
    ----------
        dt (float): Sampling time

    Methods
    -------
        motion_model(x, u): DDR motion model
        measurement_model(x): DDR measurement model
        motion_jacobian(x, u): Jacobian of motion model
        measurement_jacobian(x): Jacobian of measurement model
    """

    def __init__(self, dt):
        self.dt = dt

        super().__init__(
            f=self.motion_model,
            h=self.measurement_model,
            F_jacobian=self.motion_jacobian,
            H=self.measurement_jacobian,
            dt=dt
        )

    # -------------------------------------------------------------
    #   NONLINEAR MOTION MODEL  (f)
    # -------------------------------------------------------------
    def motion_model(self, x, u):
        """
        DDR motion model:
            px'    = px + v * cos(theta) * dt
            py'    = py + v * sin(theta) * dt
            theta' = theta + omega * dt
        """
        px, py, theta = x
        v, omega = u

        px_new = px + v * np.cos(theta) * self.dt
        py_new = py + v * np.sin(theta) * self.dt
        theta_new = theta + omega * self.dt

        return np.array([px_new, py_new, theta_new])

    # -------------------------------------------------------------
    #   NONLINEAR MEASUREMENT MODEL  (h)
    # -------------------------------------------------------------
    def measurement_model(self, x):
        """
        Example measurement: robot directly observes its position.
        z = [px, py]
        """
        px, py, _ = x
        return np.array([px, py])

    # -------------------------------------------------------------
    #   MOTION MODEL JACOBIAN  (F)
    # -------------------------------------------------------------
    def motion_jacobian(self, x, u):
        """
        Jacobian of f(x, u) w.r.t. x:

            ∂f/∂x =
            [ 1   0   -v*dt*sin(theta) ]
            [ 0   1    v*dt*cos(theta) ]
            [ 0   0          1         ]
        """
        _, _, theta = x
        v, _ = u

        F = np.array([
            [1, 0, -v * self.dt * np.sin(theta)],
            [0, 1,  v * self.dt * np.cos(theta)],
            [0, 0, 1]
        ])
        return F

    # -------------------------------------------------------------
    #   MEASUREMENT MODEL JACOBIAN  (H)
    # -------------------------------------------------------------
    def measurement_jacobian(self, x):
        """
        ∂h/∂x =
            [1 0 0]
            [0 1 0]

        Since z = [px, py].
        """
        H = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        return H
