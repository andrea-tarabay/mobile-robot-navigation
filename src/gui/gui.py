import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import queue
import time
import cv2
import numpy as np
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Ellipse

from fsm import Fsm  # your existing FSM thread class
from computer_vision.camera_capture_thread import CameraCaptureThread
from computer_vision.vision_params_manager import VisionParamsManager
from gui.init_vision_wizard import InitVisionWizard


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

        # Threads / queues
        self.camera_frame_queue = queue.Queue(maxsize=1)  # latest frame only
        self.camera_thread: CameraCaptureThread | None = None

        self.thread_fsm: Fsm | None = None

        # UI layout
        self._build_ui()

        # Polling loops
        self._schedule_camera_poll()
        # Note: we don't need a separate EKF polling loop because Fsm will call GUI via after()

        # Ensure clean shutdown
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------------------------
    # UI construction
    # -------------------------
    def _build_ui(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(side="top", fill="x", padx=8, pady=8)

        # Status label
        self.status_label = ttk.Label(self, text="Ready", font=("TkDefaultFont", 14))
        self.status_label.pack(padx=10, pady=6)

        # Buttons
        self.init_btn = ttk.Button(top_frame, text="Initialize CV", command=self.on_initialize)
        self.init_btn.pack(side="left", padx=(0, 6))

        self.start_btn = ttk.Button(top_frame, text="Start", command=self.on_start)
        self.start_btn.pack(side="left", padx=(0, 6))

        self.pause_btn = ttk.Button(top_frame, text="Pause", command=self.on_pause)
        self.pause_btn.pack(side="left", padx=(0, 6))

        self.resume_btn = ttk.Button(top_frame, text="Resume", command=self.on_resume)
        self.resume_btn.pack(side="left")

        self.stop_btn = ttk.Button(top_frame, text="Stop", command=self.on_stop)
        self.stop_btn.pack(side="left")

        # Button state at startup
        self._set_buttons_wait_for_init()

        # CV image frame (above plot)
        cv_frame = ttk.Frame(self)
        cv_frame.pack(side="top", fill="x", expand=False, padx=8, pady=(0, 8))
        self.cv_image_label = tk.Label(cv_frame)
        self.cv_image_label.pack()

        # Matplotlib plot frame (fills remainder)
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))

        # Figure & Axes
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._setup_plot_axes()

        # embed matplotlib canvas
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.mpl_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_widget.pack(side="top", fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.mpl_canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side="top", fill="x")

    # -------------------------
    # Buttons state helpers
    # -------------------------
    def _set_buttons_wait_for_init(self):
        """At application start:"""
        self.init_btn.config(state="normal")
        self._check_params_file()
        self.stop_btn.config(state="disabled")
        self.pause_btn.config(state="disabled")
        self.resume_btn.config(state="disabled")

    def _set_buttons_ready(self):
        """At application start: Start enabled, others disabled."""
        self.init_btn.config(state="normal")
        self._check_params_file()
        self.stop_btn.config(state="disabled")
        self.pause_btn.config(state="disabled")
        self.resume_btn.config(state="disabled")

    def _set_buttons_running(self):
        self.init_btn.config(state="disabled")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.pause_btn.config(state="normal")
        self.resume_btn.config(state="disabled")

    def _set_buttons_paused(self):
        self.init_btn.config(state="disabled")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.pause_btn.config(state="disabled")
        self.resume_btn.config(state="normal")

    def _set_buttons_stopped(self):
        self._set_buttons_ready()

    # -------------------------
    # Check JSON
    # -------------------------
    def _check_params_file(self):
        """Enable start button if params file exists, otherwise force initialization."""
        params_file = VisionParamsManager().path
        if os.path.exists(params_file):
            self.start_btn.config(state="normal")
            self.status_label.config(text="Ready to start")
        else:
            self.start_btn.config(state="disabled")
            self.status_label.config(text="Initialization required (no parameters found)")

    # -------------------------
    # Button callbacks
    # -------------------------
    def on_initialize(self):
        """Start computer vision initialization."""
        self.init_btn.config(state="disabled")
        self.status_label.config(text="Opening CV initialization wizard...")

        # 1. Capture ONE frame directly from the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_label.config(text="Camera not found.")
            self.init_btn.config(state="normal")
            return

        # Warm-up: try to grab a valid frame within 2 seconds
        start = time.time()
        frame = None
        while time.time() - start < 2.0:
            ok, f = cap.read()
            if ok and f is not None:
                # reject dark/black frames
                if f.mean() > 5:    # more robust than sum != 0
                    frame = f
                    break
            time.sleep(0.03)

        cap.release()

        if frame is None:
            self.status_label.config(text="Failed to capture a valid frame.")
            self.init_btn.config(state="normal")
            return

        # 2. Load parameters manager
        params = VisionParamsManager()

        # 3. Pass the captured FRAME to the wizard
        wizard = InitVisionWizard(self, frame, params)  # <-- FIXED

        wizard.grab_set()           # modal window
        self.wait_window(wizard)    # wait for wizard to finish

        # 4. Show updated UI
        self.status_label.config(text="CV Initialized!")
        self._set_buttons_ready()

    def on_start(self):
        """Start both the camera capture and the FSM worker."""
        # start camera thread (recreate each time)
        if self.camera_thread is None or not self.camera_thread.is_alive():
            self.camera_thread = CameraCaptureThread(frame_queue=self.camera_frame_queue, device_index=0)
            self.camera_thread.start()
            print("Camera capture started.")

        # start fsm thread (recreate each time)
        if self.thread_fsm is None or not self.thread_fsm.is_alive():
            # pass the safe GUI-scheduling callback
            safe_cb = self._make_fsm_gui_callback()
            self.thread_fsm = Fsm(update_callback=safe_cb)
            self.thread_fsm.start()
            print("FSM started.")

        self._set_buttons_running()
        self.status_label.config(text="Running")

    def on_stop(self):
        """Stop camera capture and FSM worker (if running)."""
        # stop FSM
        if self.thread_fsm is not None:
            try:
                self.thread_fsm.stop()
            except Exception:
                pass
            self.thread_fsm = None

        # stop camera
        if self.camera_thread is not None:
            try:
                self.camera_thread.stop()
                # optionally join briefly
                self.camera_thread.join(timeout=1.0)
            except Exception:
                pass
            self.camera_thread = None

        self._set_buttons_stopped()
        self.status_label.config(text="Stopped")

    def on_pause(self):
        if self.thread_fsm is not None:
            self.thread_fsm.pause()
            self._set_buttons_paused()
            self.status_label.config(text="Paused")

    def on_resume(self):
        if self.thread_fsm is not None:
            self.thread_fsm.resume()
            self._set_buttons_running()
            self.status_label.config(text="Running")

    # -------------------------
    # Clean shutdown
    # -------------------------
    def _on_close(self):
        """Make sure background threads are stopped before exit."""
        self.on_stop()
        # small delay to let threads shut down
        time.sleep(0.1)
        self.destroy()


    # -------------------------
    # Camera / Frame polling
    # -------------------------
    def _schedule_camera_poll(self):
        """Schedule the periodic GUI poll to read the latest camera frame from queue."""
        try:
            frame = self.camera_frame_queue.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None:
            # convert BGR -> RGB, make PhotoImage and update label (main thread only)
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                # resize to a fixed display size or keep original
                pil_img.thumbnail((640, 480))
                self._cv_photoimage = ImageTk.PhotoImage(pil_img)  # keep reference
                self.cv_image_label.config(image=self._cv_photoimage)
            except Exception as e:
                # log and ignore; do not crash GUI
                print("Error displaying frame:", e)

        # schedule next poll
        self.after(30, self._schedule_camera_poll)

    # -------------------------
    # Matplotlib helpers
    # -------------------------
    def _setup_plot_axes(self):
        self.ax.clear()
        self.ax.set_title("EKF Mean & Covariance")
        self.ax.set_xlabel("x (pixels)")
        self.ax.set_ylabel("y (pixels, downward)")

        # default limits (customize)
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(100, 0)  # y inverted => top-left origin

        self.ax.grid(True)

    def _draw_ekf(self, mean_state, cov):
        """
        Draw the EKF mean (red dot), covariance ellipse, and orientation arrow.
        This must be called from the main thread.
        """
        self.ax.clear()
        self._setup_plot_axes()

        if mean_state is None:
            self.mpl_canvas.draw_idle()
            return

        x, y, theta = mean_state

        # Plot mean
        self.ax.plot(x, y, "ro", markersize=6, label="EKF mean")

        # Orientation arrow
        arrow_length = 10.0
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)
        self.ax.arrow(x, y, dx, dy,
                      head_width=3.0, head_length=4.0,
                      fc="green", ec="green", linewidth=2, length_includes_head=True, label="theta")

        # Covariance ellipse (2x2 from top-left corner of P)
        cov_xy = cov[0:2, 0:2] if cov is not None else np.eye(2) * 1e-3

        try:
            eig_vals, eig_vecs = np.linalg.eigh(cov_xy)
        except np.linalg.LinAlgError:
            eig_vals = np.array([1e-6, 1e-6])
            eig_vecs = np.eye(2)

        eig_vals = np.maximum(eig_vals, 1e-8)

        # choose order: largest first
        order = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[order]
        eig_vecs = eig_vecs[:, order]

        n_sigma = 3.0
        width = 2 * n_sigma * np.sqrt(eig_vals[0])
        height = 2 * n_sigma * np.sqrt(eig_vals[1])

        angle = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))

        ellipse = Ellipse((x, y), width=width, height=height, angle=angle,
                          edgecolor="blue", facecolor="none", linewidth=2, alpha=0.7, label="cov")
        self.ax.add_patch(ellipse)

        self.ax.legend(loc="upper left")
        self.mpl_canvas.draw_idle()

    # -------------------------
    # FSM <-> GUI safe callback wrapper
    # -------------------------
    def _make_fsm_gui_callback(self):
        """
        Return a function to pass to Fsm that schedules _draw_ekf on the main thread.
        Fsm will call this function from its thread: we then call self.after to
        run the actual drawing on the Tk main thread.
        """
        def _cb(mean_state, cov):
            # mean_state and cov must be serializable / safe to pass
            self.after(0, lambda: self._draw_ekf(mean_state, cov))
        return _cb