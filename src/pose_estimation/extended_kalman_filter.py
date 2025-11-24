import numpy as np
from src.pose_estimation.nonlinear_system import NonlinearSystem

class ExtendedKalmanFilter:
    """
    Generic Extended Kalman Filter (EKF) implementation.

    Theoretical requirements of the system to be supplied through
    a NonlinearSystem instance:
        - nonlinear motion model      g(x, u)
        - nonlinear measurement model h(x)
        - motion model Jacobian       G(x)
        - measurement Jacobian        H(x)
        - noise covariances           Q, R

    Attributes
    ----------
    mu: np.ndarray
        State estimate vector (n×1)
    Sigma: np.ndarray
        Covariance estimate matrix (n×n)
    system: NonlinearSystem
        Nonlinear system model containing g, h, G, H, Q, R

    Methods
    -------
    predict(u): 
        Perform the EKF prediction step.
    update(z): 
        Perform the EKF update step.
    step(u, z): 
        Convenience function that performs predict + update.
    """

    def __init__(self, mu0, Sigma0, system: NonlinearSystem):
        """
        Initialize the EKF.

        Parameters
        ----------
            mu0: np.ndarray
                Initial state vector (n×1)
            Sigma0: np.ndarray
                Initial covariance matrix (n×n)
            system: NonlinearSystem
                Nonlinear system model containing g, h, G, H, Q, R
        """
        self.mu = mu0
        self.Sigma = Sigma0

        self.system = system

    # ---------------------------------------------------------------
    #                   PREDICTION STEP
    # ---------------------------------------------------------------

    def predict(self, u):
        """
        Perform the EKF prediction step.
        
        Parameters
        ----------
            u: np.ndarray
                Control input vector
        
        Returns
        -------
            mu_pred: np.ndarray
                Predicted state vector
            Sigma_pred: np.ndarray
                Predicted covariance matrix
        """
        mu_pred = self.system.predict_next_state(self.mu, u)

        G = self.system.motion_model_jac(self.mu)
        Sigma_pred = G @ self.Sigma @ G.T + self.system.get_process_noise_cov()

        self.mu = mu_pred
        self.Sigma = Sigma_pred

        return self.mu, self.Sigma

    # ---------------------------------------------------------------
    #                   UPDATE STEP
    # ---------------------------------------------------------------

    def update(self, z):
        """
        Perform the EKF update step.
        
        Parameters
        ----------
            z: np.ndarray
                Measurement vector
        
        Returns
        -------
            mu_upd: np.ndarray
                Updated state vector
            Sigma_upd: np.ndarray
                Updated covariance matrix
        """
        z_pred = self.system.predict_measurement(self.mu)
        i = z - z_pred

        H = self.system.measurement_model_jac(self.mu)
        S = H @ self.Sigma @ H.T + self.system.get_measurement_noise_cov()

        K = self.Sigma @ H.T @ np.linalg.inv(S) # TODO: use solve for numerical stability

        # Update state mean
        mu_new = self.mu + K @ i

        # Update covariance using Joseph form for numerical stability
        I = np.eye(self.Sigma.shape[0]) # TODO: understand if this is necessary
        Sigma_new = (I - K @ H) @ self.Sigma @ (I - K @ H).T + K @ self.system.get_measurement_noise_cov() @ K.T

        self.mu = mu_new
        self.Sigma = Sigma_new

        return self.mu, self.Sigma

    # ---------------------------------------------------------------
    #                   FULL EKF STEP
    # ---------------------------------------------------------------

    def step(self, u, z):
        """
        Convenience function that performs predict + update.
        
        Parameters
        ----------
            u: np.ndarray
                Control input vector
            z: np.ndarray
                Measurement vector
        
        Returns
        -------
            mu: np.ndarray
                Updated state vector after predict and update
            Sigma: np.ndarray
                Updated covariance matrix after predict and update
        """
        self.predict(u)
        return self.update(z)