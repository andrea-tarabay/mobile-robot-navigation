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
from utils.camera_utils import find_available_camera
from computer_vision.computer_vision import ComputerVisionCore

# --------------------------
# Constants
# --------------------------
IMG_WIDTH = 1920
IMG_HEIGHT = 1080


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

        # ----------------------------------------
        # THREAD SAFE QUEUES
        # ----------------------------------------
        self.cv_queue = queue.Queue(maxsize=1)  # latest frame only
        self.fsm_queue = queue.Queue()

        self.camera_thread: CameraCaptureThread | None = None
        self.thread_fsm: Fsm | None = None

        # UI layout
        self._build_ui()

        # Polling loops
        self._schedule_camera_poll()
        self._schedule_fsm_ui_poll()

        # Ensure clean shutdown
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==========================================================================
    # BUILD UI
    # ==========================================================================
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

        # Widgets for FSM data (NEW)
        self.lbl_pose = ttk.Label(self, text="Pose [mm]: ---")
        self.lbl_pose.pack()

        self.lbl_kidnapped = ttk.Label(self, text="Kidnapped: ---")
        self.lbl_kidnapped.pack()

        self.lbl_goal = ttk.Label(self, text="Goal [mm]: ---")
        self.lbl_goal.pack()

        # ------------------------
        # Camera display
        cv_frame = ttk.Frame(self)
        cv_frame.pack(side="top", fill="x", expand=False, padx=8, pady=(0, 8))
        self.cv_image_label = tk.Label(cv_frame)
        self.cv_image_label.pack()

    # ==========================================================================
    # BUTTON STATE HELPERS
    # ==========================================================================
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

    # ==========================================================================
    # INIT
    # ==========================================================================
    def _check_params_file(self):
        """Enable start button if params file exists, otherwise force initialization."""
        params_file = VisionParamsManager().path
        if os.path.exists(params_file):
            self.start_btn.config(state="normal")
            self.status_label.config(text="Ready to start")
        else:
            self.start_btn.config(state="disabled")
            self.status_label.config(text="Initialization required (no parameters found)")

    # ==========================================================================
    # BUTTON CALLBACKS
    # ==========================================================================
    def on_initialize(self):
        """Start computer vision initialization."""
        self.init_btn.config(state="disabled")
        self.status_label.config(text="Opening CV initialization wizard...")

        # 1. Capture ONE frame directly from the camera
        _, cap = find_available_camera()

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

        # 3. Pass the captured FRAME to the wizard
        wizard = InitVisionWizard(self, frame)  # <-- FIXED

        wizard.grab_set()           # modal window
        self.wait_window(wizard)    # wait for wizard to finish

        # 4. Show updated UI
        self.status_label.config(text="CV Initialized!")
        self._set_buttons_ready()

    def on_start(self):
        """Start both the camera capture and the FSM worker."""
        # start camera thread (recreate each time)
        if self.camera_thread is None or not self.camera_thread.is_alive():
            self.camera_thread = CameraCaptureThread(queue=self.cv_queue, device_index=0)
            self.camera_thread.start()
            print("Camera capture started.")

        # start fsm thread (recreate each time)
        if self.thread_fsm is None or not self.thread_fsm.is_alive():
            # pass the safe GUI-scheduling callback
            safe_cb = self._make_fsm_gui_callback()
            self.thread_fsm = Fsm(data_queue=self.cv_queue, ui_callback=safe_cb)
            self.thread_fsm.start()
            print("FSM started.")

        self._set_buttons_running()
        self.status_label.config(text="Running")

    def on_stop(self):
        """Stop camera capture and FSM worker (if running)."""
        # Stop FSM thread
        print("[GUI] Signaling threads to stop...")
        try:
            if self.thread_fsm and self.thread_fsm.is_alive():
                self.thread_fsm.stop()     # just sets flags, does NOT block
        except Exception as e:
            print("[GUI] Error stopping FSM:", e)

        # Stop camera thread
        try:
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.stop()     # just sets flags, does NOT block
        except Exception as e:
            print("[GUI] Error stopping Camera thread:", e)

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

    # ==========================================================================
    # SHUTDOWN
    # ==========================================================================
    def _on_close(self):
        """Make sure background threads are stopped before exit."""
        self.on_stop()

        # wait for camera
        if self.camera_thread and self.camera_thread.is_alive():
            print("[GUI] Waiting for camera thread...")
            self.camera_thread.join(timeout=2.0)

        # wait for FSM
        if self.thread_fsm and self.thread_fsm.is_alive():
            print("[GUI] Waiting for FSM thread...")
            self.thread_fsm.join(timeout=2.0)
        self.destroy()


    # ==========================================================================
    # Camera / Frame polling
    # ==========================================================================
    def _schedule_camera_poll(self):
        """Display latest camera frame with FSM overlays."""
        # Get latest camera frame
        try:
            cv_frame = self.cv_queue.get_nowait()
            self._last_cv_frame = cv_frame
        except queue.Empty:
            cv_frame = getattr(self, "_last_cv_frame", None)

        # Get last FSM packet
        fsm_packet = getattr(self, "_last_fsm_packet", None)

        if cv_frame is not None:
            frame = cv_frame["frame"].copy()

            # Draw path from FSM if available
            if fsm_packet is not None:
                path = fsm_packet.get("path", None)
                if path:
                    # Convert path coordinates to integers
                    pts = [tuple(map(int, p)) for p in path]
                    for i in range(len(pts) - 1):
                        cv2.line(frame, pts[i], pts[i+1], (0, 0, 255), 2)

                # Draw Extended Kalman Filter mean, orientation and covariance
                self._draw_EKF(
                    frame,
                    fsm_packet.get("pose"),
                    fsm_packet.get("pose_cov")
                )

                # Draw goal
                goal = fsm_packet.get("goal", None)
                if goal is not None:
                    goal_px = ComputerVisionCore.mm_to_px(goal)
                    cv2.circle(frame, (int(goal_px[0]), int(goal_px[1])), 6, (255, 255, 0), -1)

            # Convert to Tk image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            pil_img.thumbnail((640, 480))
            self._cv_photoimage = ImageTk.PhotoImage(pil_img)
            self.cv_image_label.config(image=self._cv_photoimage)

        # Schedule next poll
        self.after(30, self._schedule_camera_poll)



    # ==========================================================================
    # FSM → GUI POLLING (NEW)
    # ==========================================================================
    def _schedule_fsm_ui_poll(self):
        """
        Read UI updates pushed by FSM and update widgets.
        Runs on main thread via Tkinter after().
        """
        try:
            while True:
                packet = self.fsm_queue.get_nowait()
                self._handle_label_updates(packet)
        except queue.Empty:
            pass

        # poll at ~20 Hz
        self.after(50, self._schedule_fsm_ui_poll)

    # ==========================================================================
    # Handle FSM UI packets (NEW)
    # ==========================================================================
    def _handle_label_updates(self, packet):
        self._last_fsm_packet = packet  # store latest FSM info

        # Update UI labels
        if "pose" in packet:
            self.lbl_pose.config(text=f"Pose: {packet['pose'].round(1)} mm")
        if "kidnapped" in packet:
            self.lbl_kidnapped.config(text=f"Kidnapped: {packet['kidnapped']}")
        if "goal" in packet:
            self.lbl_goal.config(text=f"Goal: {packet['goal'].round(1)} mm")

    # ==========================================================================
    # Plot Extended Kalman Filter state on camera frame
    # ==========================================================================
    def _draw_EKF(self, frame, mean, cov, color=(128,0,128), n_std=2):
        """
        Draws an ellipse representing covariance on an OpenCV frame.

        mean: (x, y, theta) in mm
        cov: 3x3 covariance matrix (mm)
        color: BGR
        n_std: number of std deviations (2 = 95% confidence)
        """

        # Extract xy covariance and convert to pixels
        cov_xy = cov[:2, :2]
        cov_xy = ComputerVisionCore.mm_to_px(ComputerVisionCore.mm_to_px(cov_xy))

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov_xy)

        # Sort eigenvalues descending
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # Compute ellipse angle
        angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))

        # Compute ellipse axes (scaled by n_std)
        axis1 = np.round(n_std * np.sqrt(eigvals[0])).astype(int)
        axis2 = np.round(n_std * np.sqrt(eigvals[1])).astype(int)

        # Compute center in pixels
        center_px = (
            ComputerVisionCore.mm_to_px(mean[0]),
            ComputerVisionCore.mm_to_px(mean[1]),
        )

        # Draw ellipse
        cv2.ellipse(
            frame,
            center_px,
            (axis1, axis2),
            angle,
            0, 360,
            color,
            2
        )

        # Draw orientation arrow
        arrow_length = 50  # pixels
        theta = mean[2]
        dx = int(arrow_length * np.cos(theta))
        dy = int(arrow_length * np.sin(theta))
        cv2.arrowedLine(
            frame,
            center_px,
            (center_px[0] + dx, center_px[1] + dy),
            color,
            2,
            tipLength=0.7
        )

    # -------------------------
    # FSM <-> GUI safe callback wrapper
    # -------------------------
    def _make_fsm_gui_callback(self):
        """
        Return a function to pass to Fsm that schedules _draw_ekf on the main thread.
        Fsm will call this function from its thread: we then call self.after to
        run the actual drawing on the Tk main thread.
        """
        def _cb(packet: dict):
            self.fsm_queue.put(packet)

        return _cb