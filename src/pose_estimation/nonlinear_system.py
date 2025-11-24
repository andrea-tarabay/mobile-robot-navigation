import numpy as np

class NonlinearSystem:
    """
    Class representing a generic nonlinear state-space system with the following 
    properties:
        - g(x, u): nonlinear state transition function
        - h(x): nonlinear measurement function
        - G(x): Jacobian of g wrt state
        - H(x): Jacobian of h wrt state
        - Noise covariances Q, R

    Attributes
    ----------
        motion_model (callable): 
            State transition function g(x, u)
        measurement_model (callable): 
            Measurement function h(x)
        motion_jacobian (callable): 
            Jacobian of g with respect to x
        measurement_jacobian (callable): 
            Jacobian of h with respect to x
        Q: np.ndarray
                Process noise covariance
        R: np.ndarray
            Measurement noise covariance
        dt (float, optional): 
            Sampling time if needed by motion model

    Methods
    -------
        predict_state(x, u): 
            Predict next state given current state x and control u.
        predict_measurement(x): 
            Predict measurement given state x.
        motion_model_jac(x): 
            Compute Jacobian of g at state x.
        measurement_model_jac(x): 
            Compute Jacobian of h at state x.
    """

    def __init__(self, motion_model, measurement_model, motion_jacobian, measurement_jacobian, Q, R, dt=None):
        """
        Initialize the nonlinear system model.

        Parameters
        ----------
            motion_model (callable): 
                State transition function g(x, u)
            measurement_model (callable): 
                Measurement function h(x)
            motion_jacobian (callable): 
                Jacobian of g with respect to x
            measurement_jacobian (callable): 
                Jacobian of h with respect to x
            Q: np.ndarray
                Process noise covariance
            R: np.ndarray
                Measurement noise covariance
            dt (float, optional): 
                Sampling time if needed by motion model
        """
        self.motion_model = motion_model
        self.measurement_model = measurement_model

        self.motion_jacobian = motion_jacobian
        self.measurement_jacobian = measurement_jacobian

        self.Q = Q
        self.R = R

        self.dt = dt

    def predict_next_state(self, x, u):
        """
        Predict next state given current state x and control u.

        Parameters
        ----------
            x: np.ndarray
                Current state vector
            u: np.ndarray
                Control input vector

        Returns
        -------
            np.ndarray
                Predicted next state vector
        """
        return self.motion_model(x, u)

    def predict_measurement(self, x):
        """
        Predict measurement given state x
        
        Parameters
        ----------
            x: np.ndarray
                Current state vector
        
        Returns
        -------
            np.ndarray
                Predicted measurement vector
        """
        return self.measurement_model(x)

    def motion_model_jac(self, x):
        """
        Compute Jacobian of motion model at state x.
        
        Parameters
        ----------
            x: np.ndarray
                Current state vector
        
        Returns
        -------
            np.ndarray
                Jacobian matrix of motion model
        """
        return self.motion_jacobian(x)

    def measurement_model_jac(self, x):
        """
        Compute Jacobian of measurement model at state x.

        Parameters
        ----------
            x: np.ndarray
                Current state vector

        Returns
        -------
            np.ndarray
                Jacobian matrix of measurement model
        """
        return self.measurement_jacobian(x)
    
    def get_process_noise_cov(self):
        """
        Get process noise covariance matrix Q.

        Returns
        -------
            np.ndarray
                Process noise covariance matrix Q
        """
        return self.Q

    def get_measurement_noise_cov(self):
        """
        Get measurement noise covariance matrix R.

        Returns
        -------
            np.ndarray
                Measurement noise covariance matrix R
        """
        return self.R