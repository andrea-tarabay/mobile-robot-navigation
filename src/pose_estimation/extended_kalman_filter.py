import numpy as np
from src.pose_estimation.nonlinear_system import NonlinearSystem

class ExtendedKalmanFilter:
    """
    Generic Extended Kalman Filter (EKF) implementation.

    Theoretical requirements of the system to be supplied:
        - nonlinear motion model      g(x, u)
        - nonlinear measurement model h(x)
        - motion model Jacobian       G(x)
        - measurement Jacobian        H(x)
        - noise covariances           Q, R

    The internal state mean and covariance, as well as the
    system models and noise covariances, are stored in the 
    class as follows:

    Attributes
    ----------
    mu: np.ndarray
        State estimate vector (n×1)
    Sigma: np.ndarray
        Covariance estimate matrix (n×n)
    g: callable
        State transition function
    h: callable
        Measurement function
    G: callable
        Jacobian of state transition
    H: callable
        Jacobian of measurement model
    Q: np.ndarray
        Process noise covariance
    R: np.ndarray
        Measurement noise covariance

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    def __init__(self, mu0, Sigma0, system: NonlinearSystem, Q, R):
        """
        Initialize the EKF.

        Parameters
        ----------
            mu0: np.ndarray
                Initial state vector (n×1)
            Sigma0: np.ndarray
                Initial covariance matrix (n×n)
            g: callable
                State transition function
            h: callable
                Measurement function
            G: callable
                Jacobian of state transition
            H: callable
                Jacobian of measurement model
            Q: np.ndarray
                Process noise covariance
            R: np.ndarray
                Measurement noise covariance
        """
        self.mu = mu0
        self.Sigma = Sigma0

        self.system = system

        self.Q = Q
        self.R = R

    # ---------------------------------------------------------------
    #                   PREDICTION STEP
    # ---------------------------------------------------------------

    def predict(self, u):
        """Perform the EKF prediction step."""
        # Nonlinear prediction
        mu_pred = self.system.predict_next_state(self.mu, u)

        # Jacobian evaluation
        G = self.system.motion_model_jac(self.mu)

        # Covariance propagation
        Sigma_pred = G @ self.Sigma @ G.T + self.Q

        # Store
        self.mu = mu_pred
        self.Sigma = Sigma_pred

        return self.mu, self.Sigma

    # ---------------------------------------------------------------
    #                   UPDATE STEP
    # ---------------------------------------------------------------

    def update(self, z):
        """Perform the EKF update step."""
        # Predict measurement
        z_pred = self.system.predict_measurement(self.mu)

        # Measurement Jacobian
        H = self.system.measurement_model_jac(self.mu)

        # Innovation
        i = z - z_pred

        # Innovation covariance
        S = H @ self.Sigma @ H.T + self.R

        # Kalman gain
        K = self.Sigma @ H.T @ np.linalg.inv(S) # TODO: use solve for numerical stability

        # Update mean
        mu_new = self.mu + K @ i

        # Joseph form covariance update for stability
        I = np.eye(self.Sigma.shape[0]) # TODO: understand if this is necessary
        Sigma_new = (I - K @ H) @ self.Sigma @ (I - K @ H).T + K @ self.R @ K.T

        self.mu = mu_new
        self.Sigma = Sigma_new

        return self.mu, self.Sigma

    # ---------------------------------------------------------------
    #                   FULL EKF STEP
    # ---------------------------------------------------------------

    def step(self, u, z):
        """Convenience function that performs predict + update."""
        self.predict(u)
        return self.update(z)