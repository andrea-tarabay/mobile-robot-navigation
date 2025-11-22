import numpy as np

class NonlinearSystem:
    """
    Generic nonlinear system model for use with an Extended Kalman Filter.
    
    Subclasses or instances should define:
        - g(x, u): nonlinear state transition function
        - h(x): nonlinear measurement function
        - G(x): Jacobian of g wrt state
        - H(x): Jacobian of h wrt state

    Attributes
    ----------
        g (callable): State transition function g(x, u)
        h (callable): Measurement function h(x)
        G (callable): Jacobian of g with respect to x
        H (callable): Jacobian of h with respect to x
        dt (float, optional): Sampling time if needed by motion model

    Methods
    -------
        predict_state(x, u): Predict next state given current state x and control u.
        predict_measurement(x): Predict measurement given state x.
        G_jac(x): Compute Jacobian of g at state x.
        H_jac(x): Compute Jacobian of h at state x.
    """

    def __init__(self, g, h, G, H, dt=None):
        """
        Initialize the nonlinear system model.

        Parameters
        ----------
            g (callable): State transition function g(x, u)
            h (callable): Measurement function h(x)
            G (callable): Jacobian of g with respect to x
            H (callable): Jacobian of h with respect to x
            dt (float, optional): Sampling time if needed by motion model
        """
        self.g = g
        self.h = h
        self.G = G
        self.H = H
        self.dt = dt

    # Wrappers (optional, for convenience and readability)
    def predict_state(self, x, u):
        return self.g(x, u)

    def predict_measurement(self, x):
        return self.h(x)

    def G_jac(self, x):
        return self.G(x)

    def H_jac(self, x):
        return self.H(x)
