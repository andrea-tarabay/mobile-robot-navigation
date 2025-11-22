import numpy as np
from src.pose_estimation.nonlinear_system import NonlinearSystem


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
        reduced: bool
            If True, use reduced measurement model (wheel encoders only), else full model 
            (camera + wheel encoders)

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

    def __init__(self, dt, lambda_, axle_length, reduced=False):
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
            reduced: bool
                If True, use reduced measurement model (wheel encoders only), else full model
        """
        self.lambda_ = lambda_
        self.axle_length = axle_length
        self.reduced = reduced

        super().__init__(
            motion_model=self.motion_model,
            measurement_model=self.measurement_model,
            motion_jacobian=self.motion_jacobian,
            measurement_jacobian=self.measurement_jacobian,
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
            v'     = lambda_/2 * (ur + ul)
            omega' = lambda_/(axle_length) * (ur - ul)
        
        Parameters
        ----------
            x: np.ndarray
                Current state vector [px, py, theta, v, omega]
            u: np.ndarray
                Control input vector [ur, ul]
        
        Returns
        -------
            np.ndarray
                Predicted next state vector
        """
        px, py, theta, v, omega = x
        ur, ul = u

        px_new = px + self.dt * v * np.cos(theta)
        py_new = py + self.dt * v * np.sin(theta)
        theta_new = theta - self.dt * omega
        v_new = self.lambda_ * (ur + ul) / 2
        omega_new = self.lambda_ * (ur - ul) / self.axle_length

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
            mur = 1/lambda_ * (v + axle_length/2 * omega)
            mul = 1/lambda_ * (v - axle_length/2 * omega)

        If reduced=True, only return wheel measurements (no camera):
            mur = 1/lambda_ * (v + axle_length/2 * omega)
            mul = 1/lambda_ * (v - axle_length/2 * omega)
        
        Parameters
        ----------
            x: np.ndarray
                Current state vector [px, py, theta, v, omega]
        
        Returns
        -------
            np.ndarray
                Predicted measurement vector
        """
        if self.reduced:
            _, _, _, v, omega = x

            mur = 1/self.lambda_ * (v + self.axle_length/2 * omega)
            mul = 1/self.lambda_ * (v - self.axle_length/2 * omega)

            return np.array([mur, mul])
        else:
            mpx, mpy, mtheta, v, omega = x

            mur = 1/self.lambda_ * (v + self.axle_length/2 * omega)
            mul = 1/self.lambda_ * (v - self.axle_length/2 * omega)

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
        
        Parameters
        ----------
            x: np.ndarray
                Current state vector [px, py, theta, v, omega]
        
        Returns
        -------
            np.ndarray
                Jacobian matrix of the motion model
        """
        _, _, theta, v, _ = x

        G = np.array([
            [1, 0, -self.dt * v * np.sin(theta), self.dt * np.cos(theta), 0],
            [0, 1,  self.dt * v * np.cos(theta), self.dt * np.sin(theta), 0],
            [0, 0, 1, 0, -self.dt],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        return G

    # -------------------------------------------------------------
    #   MEASUREMENT MODEL JACOBIAN  (H)
    # -------------------------------------------------------------
    def measurement_jacobian(self, x):
        """
        Jacobian of h(x) w.r.t. x:
            ∂h/∂x =
            [ 1   0   0       0                   0            ]
            [ 0   1   0       0                   0            ]
            [ 0   0   1       0                   0            ]
            [ 0   0   0   1/lambda_    axle_length/(2*lambda_) ]
            [ 0   0   0   1/lambda_   -axle_length/(2*lambda_) ]

        If reduced=True, the Jacobian is:
            ∂h_reduced/∂x =
            [ 0   0   0   1/lambda_    axle_length/(2*lambda_) ]
            [ 0   0   0   1/lambda_   -axle_length/(2*lambda_) ]
        
        Parameters
        ----------
            x: np.ndarray
                Current state vector [px, py, theta, v, omega]
        
        Returns
        -------
            np.ndarray
                Jacobian matrix of the measurement model
        """
        if self.reduced:
            H_reduced = np.array([
                [0, 0, 0, 1/self.lambda_,  self.axle_length/(2*self.lambda_)],
                [0, 0, 0, 1/self.lambda_, -self.axle_length/(2*self.lambda_)]
            ])
            return H_reduced
        else:
            H = np.array([
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1/self.lambda_,  self.axle_length/(2*self.lambda_)],
                [0, 0, 0, 1/self.lambda_, -self.axle_length/(2*self.lambda_)]
            ])
            return H
