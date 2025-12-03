import numpy as np
from pose_estimation.nonlinear_system import NonlinearSystem


class DifferentialDriveSystem(NonlinearSystem):
    """
    Class representing a differential-drive robot (DDR) state-space system.

    Defines the motion and measurement models along with their Jacobians
    overriding the NonlinearSystem super class. Two versions of the measurement
    model are provided: full (camera + wheel encoders) and reduced (wheel encoders only).

    Attributes
    ----------
        dt: float
            Sampling time
        lambda_: float
            Conversion factor from wheel control input to robot linear/angular velocity
        axle_length: float
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

    def __init__(self, dt: float, lambda_: float, axle_length: float, 
                 motion_noise_cov: np.ndarray, measurement_noise_cov: np.ndarray):
        """
        Initialize the DDR system model.
        
        Parameters
        ----------
            dt: float
                Sampling time
            lambda_: float
                Conversion factor from wheel control input to robot linear/angular velocity
            axle_length: float
                Distance between the wheels
            motion_noise_cov: np.ndarray
                Process noise covariance
            measurement_noise_cov: np.ndarray
                Measurement noise covariance
        """
        self.lambda_ = lambda_
        self.axle_length = axle_length
        
        super().__init__(
            motion_model=self.motion_model,
            measurement_model=self.measurement_model,
            motion_jacobian=self.motion_jacobian,
            measurement_jacobian=self.measurement_jacobian,
            Q=motion_noise_cov,
            R=measurement_noise_cov,
            dt=dt
        )

    # -------------------------------------------------------------
    #   NONLINEAR MOTION MODEL  (g)
    # -------------------------------------------------------------
    def motion_model(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        DDR motion model:
            px'    = px + dt * lambda_/2 * (ur + ul) * cos(theta)
            py'    = py + dt * lambda_/2 * (ur + ul) * sin(theta)
            theta' = theta - dt * lambda_/(axle_length) * (ur - ul)
        
        Parameters
        ----------
            x: np.ndarray
                Current state vector [px, py, theta, v, omega]
            u: np.ndarray
                Control input vector [ur, ul], i.e. the odometry readings
        
        Returns
        -------
            np.ndarray
                Predicted next state vector
        """
        px, py, theta = x
        ur, ul = u

        px_new = px + self.dt * self.lambda_ * (ur + ul) / 2 * np.cos(theta)
        py_new = py + self.dt * self.lambda_ * (ur + ul) / 2 * np.sin(theta)
        theta_new = theta - self.dt * self.lambda_ * (ur - ul) / self.axle_length

        return np.array([px_new, py_new, theta_new])

    # -------------------------------------------------------------
    #   NONLINEAR MEASUREMENT MODEL  (h)
    # -------------------------------------------------------------
    def measurement_model(self, x: np.ndarray) -> np.ndarray:
        """
        DDR measurement model:
            mpx = px
            mpy = py
            mtheta = theta
        
        Parameters
        ----------
            x: np.ndarray
                Current state vector [px, py, theta]
        
        Returns
        -------
            np.ndarray
                Predicted measurement vector
        """
        return x           

    # -------------------------------------------------------------
    #   MOTION MODEL JACOBIAN  (G)
    # -------------------------------------------------------------
    def motion_jacobian(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Jacobian of g(x, u) w.r.t. x:
            ∂g/∂x =
            [ 1   0   -dt * lambda_/2 * (ur + ul) * sin(theta) ]
            [ 0   1    dt * lambda_/2 * (ur + ul) * cos(theta) ]
            [ 0   0                         1                  ]
        
        Parameters
        ----------
            x: np.ndarray
                Current state vector [px, py, theta]
            u: np.ndarray
                Control input vector [ur, ul], i.e. the odometry readings
        
        Returns
        -------
            np.ndarray
                Jacobian matrix of the motion model
        """
        _, _, theta = x
        ur, ul = u

        G = np.array([
            [1, 0, -self.dt * self.lambda_/2 * (ur + ul) * np.sin(theta)],
            [0, 1,  self.dt * self.lambda_/2 * (ur + ul) * np.cos(theta)],
            [0, 0, 1]
        ])
        return G

    # -------------------------------------------------------------
    #   MEASUREMENT MODEL JACOBIAN  (H)
    # -------------------------------------------------------------
    def measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of h(x) w.r.t. x:
            ∂h/∂x =
            [ 1   0   0 ]
            [ 0   1   0 ]
            [ 0   0   1 ]
        
        Parameters
        ----------
            x: np.ndarray
                Current state vector [px, py, theta]
        
        Returns
        -------
            np.ndarray
                Jacobian matrix of the measurement model
        """
        return np.eye(3)
