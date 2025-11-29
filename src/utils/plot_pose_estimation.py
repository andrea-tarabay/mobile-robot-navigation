import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def covariance_ellipse(P, n_std=3.0):
    """
    Compute ellipse parameters (width, height, angle) for a covariance matrix P.

    Args:
        P (np.ndarray): 2x2 covariance matrix
        n_std (float): Number of standard deviations (default: 3)

    Returns:
        width, height, angle (degrees)
    """
    # Eigen-decomposition
    eigenvals, eigenvecs = np.linalg.eigh(P)
    
    # Sort eigenvalues largest -> smallest
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]

    # Extract ellipse properties
    width = 2 * n_std * np.sqrt(eigenvals[0])
    height = 2 * n_std * np.sqrt(eigenvals[1])

    # Angle of ellipse (in degrees)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

    return width, height, angle


def plot_covariance_ellipse(ax, mean, P, n_std=3.0, **kwargs):
    """
    Plot a 2D covariance ellipse.

    Args:
        ax (axes): Matplotlib Axes object
        mean (np.ndarray): 2x1 state mean [x, y]
        P (np.ndarray): 2x2 covariance matrix for (x, y)
        n_std (float): Number of standard deviations

    Returns:
        Ellipse artist
    """
    width, height, angle = covariance_ellipse(P, n_std)

    ellipse = Ellipse(
        xy=(mean[0], mean[1]),
        width=width,
        height=height,
        angle=angle,
        **kwargs
    )
    ax.add_patch(ellipse)
    return ellipse


def plot_state_with_covariance(ax, x, P, n_std=3, color='blue'):
    """
    Plot a 2D state and its covariance ellipse.
    
    Args:
        ax (axes): Matplotlib axis
        x (np.ndarray): State vector (must contain x[0], x[1])
        P (np.ndarray): Covariance matrix (2x2 for xy subspace)
    """
    ax.plot(x[0], x[1], 'o', color=color)
    plot_covariance_ellipse(
        ax, x[:2], P[:2, :2],
        n_std=n_std,
        fill=False,
        edgecolor=color,
        linewidth=1.5
    )
