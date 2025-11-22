import numpy as np
from src.pose_estimation.nonlinear_system import NonlinearSystem


class DifferentialDriveSystem(NonlinearSystem):
    """
    Nonlinear system model for a differential-drive robot (DDR).

    Defines the motion and measurement models along with their Jacobians
    overriding the NonlinearSystem super class.

    Attributes
    ----------
        dt: float
            Sampling time
        lambda_: float
            Conversion factor from wheel control input to robot linear/angular velocity
        d: float
            Distance between the wheels

    Methods
    -------
        motion_model(x, u): 
            DDR motion model
        measurement_model(x): 
            DDR measurement model
        motion_jacobian(x, u): 
            Jacobian of motion model
        measurement_jacobian(x): 
            Jacobian of measurement model
    """

    def __init__(self, dt, lambda_, d):
        self.dt = dt
        self.lambda_ = lambda_
        self.d = d

        super().__init__(
            g=self.motion_model,
            h=self.measurement_model,
            G=self.motion_jacobian,
            H=self.measurement_jacobian,
            dt=dt
        )

    # -------------------------------------------------------------
    #   NONLINEAR MOTION MODEL  (g)
    # -------------------------------------------------------------
    def motion_model(self, x, u):
        """
        DDR motion model:
            px'    = px + dt * v * cos(theta)
            py'    = py + dt * v * sin(theta)
            theta' = theta - dt * omega
            v'     = lamda/2 * (ur + ul)
            omega' = lamda/(d) * (ur - ul)
        """
        px, py, theta, v, omega = x
        ur, ul = u

        px_new = px + self.dt * v * np.cos(theta)
        py_new = py + self.dt * v * np.sin(theta)
        theta_new = theta - self.dt * omega
        v_new = self.lambda_ * (ur + ul) / 2
        omega_new = self.lambda_ * (ur - ul) / self.d

        return np.array([px_new, py_new, theta_new, v_new, omega_new])

    # -------------------------------------------------------------
    #   NONLINEAR MEASUREMENT MODEL  (h)
    # -------------------------------------------------------------
    def measurement_model(self, x): # TODO: call sensors function ? NO ! It is just the model
        """
        DDR measurement model:
            mpx = px
            mpy = py
            mtheta = theta
            mur = 1/lambda_ * (v + d/2 * omega)
            mul = 1/lambda_ * (v - d/2 * omega)
        """
        mpx, mpy, mtheta, v, omega = x

        mur = 1/self.lambda_ * (v + self.d/2 * omega)
        mul = 1/self.lambda_ * (v - self.d/2 * omega)
        
        return np.array([mpx, mpy, mtheta, mur, mul])

    # -------------------------------------------------------------
    #   MOTION MODEL JACOBIAN  (G)
    # -------------------------------------------------------------
    def motion_jacobian(self, x):
        """
        Jacobian of g(x, u) w.r.t. x:

            ∂g/∂x =
            [ 1   0   -dt*v*sin(theta)   dt*cos(theta)   0 ]
            [ 0   1    dt*v*cos(theta)   dt*sin(theta)   0 ]
            [ 0   0          1                 0       -dt ]
            [ 0   0          0                 0         0 ]
            [ 0   0          0                 0         0 ]
        """
        _, _, theta, v, _ = x

        F = np.array([
            [1, 0, -self.dt * v * np.sin(theta), self.dt * np.cos(theta), 0],
            [0, 1,  self.dt * v * np.cos(theta), self.dt * np.sin(theta), 0],
            [0, 0, 1, 0, -self.dt],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        return F

    # -------------------------------------------------------------
    #   MEASUREMENT MODEL JACOBIAN  (H)
    # -------------------------------------------------------------
    def measurement_jacobian(self, x): # TODO: add the second jacobian in case cam not available !
        """
        Jacobian of h(x) w.r.t. x:

            ∂h/∂x =
            [ 1   0   0       0              0       ]
            [ 0   1   0       0              0       ]
            [ 0   0   1       0              0       ]
            [ 0   0   0   1/lambda_    d/(2*lambda_) ]
            [ 0   0   0   1/lambda_   -d/(2*lambda_) ]
        """
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1/self.lambda_,  self.d/(2*self.lambda_)],
            [0, 0, 0, 1/self.lambda_, -self.d/(2*self.lambda_)]
        ])
        return H
