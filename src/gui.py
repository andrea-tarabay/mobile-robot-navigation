import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Ellipse

from fsm import Fsm  # Your worker thread class


class Gui(tk.Tk):
    """
    Tkinter GUI for displaying EKF results (mean + covariance ellipse) and
    controlling a background FSM thread.

    The FSM is expected to push EKF outputs (mean, covariance) into a
    thread-safe queue. This GUI periodically checks the queue and updates the plot.
    """

    def __init__(self, update_rate_ms: int = 100):
        """
        Initialize the GUI window, widgets, matplotlib plot, and FSM thread.

        Parameters
        ----------
        update_rate_ms : int
            How often the GUI checks for new EKF results in the queue (ms).
        """
        super().__init__()
        self.title("Thymio on its Treasure Hunt")
        self.geometry("900x650")

        self.update_rate_ms = update_rate_ms  # refresh rate for plot updates

        # --------------------------- Thread Management ---------------------------- #

        # Fsm thread must expose a thread-safe queue named ekf_queue
        self.thread = None

        # --------------------------- UI Elements --------------------------------- #

        # --- Top frame with buttons ---
        top_frame = ttk.Frame(self)
        top_frame.pack(side="top", fill="x", padx=8, pady=8)

        self.start_btn = ttk.Button(top_frame, text="Start", command=self.on_start)
        self.start_btn.pack(side="left", padx=(0, 6))

        self.pause_btn = ttk.Button(top_frame, text="Pause", command=self.on_pause)
        self.pause_btn.pack(side="left", padx=(0, 6))

        self.resume_btn = ttk.Button(top_frame, text="Resume", command=self.on_resume)
        self.resume_btn.pack(side="left")

        self.stop_btn = ttk.Button(top_frame, text="Stop", command=self.on_stop)
        self.stop_btn.pack(side="left")

        # Set initial button state
        self._set_buttons_initial()

        # Display latest EKF text output
        self.label = ttk.Label(self, text="Waiting for data...", font=("TkDefaultFont", 18))
        self.label.pack(padx=10, pady=10)

        # --- Plot frame (fills the rest) ---
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))

        # Matplotlib Figure
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._setup_plot()

        # Embed the matplotlib figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="top", fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side="top", fill="x")

    # -------------------------------------------------------------------------- #
    #                             Button Callbacks                               #
    # -------------------------------------------------------------------------- #

    def on_start(self):
        """Start the FSM worker thread."""
        if self.thread is None or not self.thread.is_alive():
            self.thread = Fsm(update_callback=self._update_ekf_plot)   # <-- create a new one
            self.thread.start()
            print("Thread started.")
            self._set_buttons_running()
        else:
            print("Thread already running.")

    def on_stop(self):
        """Signal the FSM thread to stop."""
        if self.thread and self.thread.is_alive():
            self.thread.stop()
            print("Stop signal sent.")
        else:
            print("No running thread to stop.")
        self._set_buttons_stopped()

    def on_pause(self):
        if self.thread and self.thread.is_alive():
            self.thread.pause()
            print("Paused.")
            self._set_buttons_paused()

    def on_resume(self):
        if self.thread and self.thread.is_alive():
            self.thread.resume()
            print("Resumed.")
            self._set_buttons_running()

    # --------------------------- Button state logic ---------------------------- #

    def _set_buttons_initial(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.pause_btn.config(state="disabled")
        self.resume_btn.config(state="disabled")

    def _set_buttons_running(self):
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.pause_btn.config(state="normal")
        self.resume_btn.config(state="disabled")

    def _set_buttons_paused(self):
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.pause_btn.config(state="disabled")
        self.resume_btn.config(state="normal")

    def _set_buttons_stopped(self):
        self._set_buttons_initial()

    # -------------------------------------------------------------------------- #
    #                              Matplotlib Setup                               #
    # -------------------------------------------------------------------------- #

    def _setup_plot(self):
        """Initial configuration of the plot area."""
        self.ax.clear()
        self.ax.set_title("EKF Mean & Covariance")
        self.ax.set_xlabel("x (pixels)")
        self.ax.set_ylabel("y (pixels, downward)")
        self.ax.set_xlim(0, 500)
        self.ax.set_ylim(0, 500)

        # Make origin top-left like an image
        self.ax.invert_yaxis()

        self.ax.grid(True)

    # -------------------------------------------------------------------------- #
    #                         EKF Plotting (mean + covariance)                   #
    # -------------------------------------------------------------------------- #

    def _update_ekf_plot(self, mean_state, cov):
        """
        Draw EKF mean as a point and covariance as an ellipse.

        Parameters
        ----------
        mean : np.ndarray
            2-element mean vector [x, y]
        cov : np.ndarray
            2x2 covariance matrix
        """
        self.ax.clear()
        self._setup_plot()  # redraw titles, axes, grid

        x, y, theta = mean_state

        # --- Plot mean as a small dot ---
        self.ax.plot(x, y, "ro", markersize=5, label="EKF Mean")

        # --- Plot heading as an arrow ---
        arrow_length = 15.0   # adjust as needed

        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)

        self.ax.arrow(
            x, y, dx, dy,
            head_width=2.0,
            head_length=3.0,
            fc='green',
            ec='green',
            linewidth=2,
            length_includes_head=True,
            label="Theta"
        )

        # --- Compute covariance ellipse ---
        cov_xy = cov[0:2, 0:2]     # 2x2 position covariance

        # Ensure covariance is PSD
        try:
            eig_vals, eig_vecs = np.linalg.eigh(cov_xy)
        except np.linalg.LinAlgError:
            eig_vals = np.array([1e-6, 1e-6])
            eig_vecs = np.eye(2)
        
        # clamp eigenvalues to avoid negative / zero / tiny
        eig_vals = np.maximum(eig_vals, 1e-6)

        # Convert to ellipse axes lengths
        # scale = n-sigma confidence (n=3 for 99.7%)
        n_sigma = 3.0
        width = 2 * n_sigma * np.sqrt(eig_vals[0])
        height = 2 * n_sigma * np.sqrt(eig_vals[1])

        # Ellipse orientation
        angle_deg = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))

        ellipse = Ellipse(
            xy=(x, y),
            width=width,
            height=height,
            angle=angle_deg,
            edgecolor="blue",
            facecolor="none",
            linewidth=2,
            alpha=0.7,   # <-- transparency
            label="Covariance",
        )
        self.ax.add_patch(ellipse)

        self.ax.legend()
        self.canvas.draw()