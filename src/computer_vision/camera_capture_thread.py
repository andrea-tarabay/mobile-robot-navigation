import platform
import threading
import queue
import time
import cv2
import numpy as np
from shapely.geometry import Polygon

from computer_vision.vision_params_manager import VisionParamsManager
from computer_vision.computer_vision import ComputerVisionCore

class CameraCaptureThread(threading.Thread):
    """
    Background thread that continuously captures frames from a camera device
    and pushes the latest frame to a queue.Queue(maxsize=1). It never touches Tk.
    """
    def __init__(self, queue: queue.Queue, device_index: int = 0, warmup_timeout: float = 2.0):
        super().__init__(daemon=True)
        self.queue = queue
        self.device_index = device_index
        self._stop_event = threading.Event()
        self.warmup_timeout = warmup_timeout

        # Vision parameters manager
        self.params_manager = VisionParamsManager()
        self.params_manager.load()

        # Store precomputed obstacles and goal position
        self.static_obstacles: list[Polygon] = []
        self.goal = None  # detected goal position

        # EMA filter state variables
        self.smoothed_center = None
        self.smoothed_red = None
        self.smoothed_green = None
        self.smoothed_theta = None

        # Safe lock for robot pose
        self.robot_pose_lock = threading.Lock()
        self.robot_pose = None  # last smoothed robot pose


    def stop(self):
        self._stop_event.set()

    def _grab_first_valid_frame(self, cap, timeout):
        """
        Try to grab a valid (non-black) frame within timeout seconds.
        Returns frame or None.
        """
        start = time.time()
        frame = None
        while time.time() - start < timeout and not self._stop_event.is_set():
            ok, frame = cap.read()
            if ok and frame is not None:
                # heuristics: not pure black
                if frame.sum() != 0:
                    return frame
            time.sleep(0.05)
        return None
    
    # -------------------------------
    # Utility: EMA for 2D points
    # -------------------------------
    def _smooth_point(self, prev, new, alpha: float):
        """Exponential smoothing for 2D points."""
        if new is None:
            return prev  # keep previous
        if prev is None:
            return np.array(new, dtype=float)
        return alpha * np.array(new) + (1 - alpha) * prev

    # -------------------------------
    # Utility: EMA for angle
    # -------------------------------
    def _smooth_angle(self, prev, new, alpha: float):
        """Exponential smoothing for angles, avoiding wrap-around jumps."""
        if new is None:
            return prev
        if prev is None:
            return new
        # shortest angular difference
        diff = np.arctan2(np.sin(new - prev), np.cos(new - prev))
        return prev + alpha * diff

    # -------------------------------
    # Apply EMA to raw robot detection
    # -------------------------------
    def smooth_robot_pose(self, robot, alpha: float = 0.6):
        """Apply EMA smoothing to raw robot detection."""
        if not robot["found"]:
            return None  # No detection, skip

        # Smooth each attribute
        self.smoothed_center = self._smooth_point(self.smoothed_center, robot["center"], alpha)
        self.smoothed_red    = self._smooth_point(self.smoothed_red, robot["red_center"], alpha)
        self.smoothed_green  = self._smooth_point(self.smoothed_green, robot["green_center"], alpha)
        self.smoothed_theta  = self._smooth_angle(self.smoothed_theta, robot["theta"], alpha)

        # Build smoothed pose
        smoothed = {
            "found": True,
            "center": (int(round(self.smoothed_center[0])), int(round(self.smoothed_center[1]))) if self.smoothed_center is not None else robot["center"],
            "red_center": (int(round(self.smoothed_red[0])), int(round(self.smoothed_red[1]))) if self.smoothed_red is not None else robot["red_center"],
            "green_center": (int(round(self.smoothed_green[0])), int(round(self.smoothed_green[1]))) if self.smoothed_green is not None else robot["green_center"],
            "theta": self.smoothed_theta,
        }

        return smoothed

    def run(self):
        cap = self.open_camera(self.device_index)

        # Initialize static obstacles
        self.init_obstacles_and_goal(cap)

        # Continuous capture loop
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                # small sleep to avoid busy loop if camera fails temporarily
                time.sleep(0.05)
                continue

            # Process frame:
            processed_frame = frame.copy()
            smoothed_robot, processed_frame = self.process_frame(processed_frame)

            # Compute global path

            # Push latest frame into queue, keep only newest frame (maxsize=1)
            self.push_processed_data(processed_frame, smoothed_robot)

            # throttle capture rate slightly to reduce CPU (adjust as needed)
            time.sleep(0.01)

        cap.release()

    def open_camera(self, device_index: int):
        system = platform.system()
        
        if system == "Darwin":
            # macOS: use AVFoundation
            cap = cv2.VideoCapture(device_index)
        elif system == "Windows":
            # Windows: use DirectShow and add a device offset
            cap = cv2.VideoCapture(device_index + 1, cv2.CAP_DSHOW)
        else:
            # Linux / default
            cap = cv2.VideoCapture(device_index)
        
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera index {device_index} on {system}")
        
        return cap


    def init_obstacles_and_goal(self, cap):
        first_frame = self._grab_first_valid_frame(cap, self.warmup_timeout)
        if first_frame is not None:
            # Detect goal position
            self.goal = ComputerVisionCore.detect_goal(first_frame, self.params_manager.color_params)

            # Precompute static obstacles
            self.static_obstacles = ComputerVisionCore.detect_obstacles(
                        first_frame, self.params_manager.color_params, 
                        self.params_manager.canny_params, 
                        self.params_manager.poly_params
                    )

    def process_frame(self, frame: np.ndarray):
        """
        Process incoming frame from the queue.
        Returns dict with keys: frame (np.ndarray), robot (dict)
        or None if no frame is available.
        """
        overlay_frame = frame.copy()

        # --- Raw robot detection ---
        raw_robot = ComputerVisionCore.detect_robot(frame, self.params_manager.color_params)

        # --- Apply smoothing ---
        smoothed_robot = self.smooth_robot_pose(raw_robot)

        # Store smoothed pose for other threads
        with self.robot_pose_lock:
            self.robot_pose = smoothed_robot

        # Overlay static obstacles on the frame
        if self.static_obstacles:
            for poly in self.static_obstacles:
                pts = np.array(poly.exterior.coords, np.int32)   # Nx2 array
                cv2.polylines(overlay_frame, [pts], isClosed=True, color=(0,255,0), thickness=2)

        # Overlay goal position
        if self.goal and self.goal.get("found") and self.goal.get("center") is not None:
            cv2.circle(overlay_frame, self.goal["center"], 5, (0,255,255), -1)

        # Draw robot markers if found
        if smoothed_robot is not None and smoothed_robot.get("found"):
            center = smoothed_robot.get("center")
            theta = smoothed_robot.get("theta")  # radians

            arrow_length = 60  # pixels
            dx = int(arrow_length * np.cos(theta))
            dy = int(arrow_length * np.sin(theta))

            # Red marker
            cv2.circle(overlay_frame, center, 5, (0,0,255), -1)

            # Orientation arrow
            cv2.arrowedLine(
                overlay_frame,
                center,
                (center[0] + dx, center[1] + dy),
                color=(255,255,0),  # cyan
                thickness=2,
                tipLength=0.7
            )
        
        return smoothed_robot, overlay_frame

    def push_processed_data(self, frame: np.ndarray, robot: dict):
        """
        Push processed frame and robot data into the queue.
        """
        try:
            self.queue.put_nowait({"frame": frame, "robot": robot})
        except queue.Full:
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait({"frame": frame, "robot": robot})
            except queue.Full:
                pass
    
    # -------------------------------
    # Access robot pose safely
    # -------------------------------
    def get_robot_pose(self):
        """Thread-safe access to the current smoothed robot pose."""
        with self.robot_pose_lock:
            return self.robot_pose.copy() if self.robot_pose else None