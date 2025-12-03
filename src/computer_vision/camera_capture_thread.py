import threading
import queue
import time
import cv2
import numpy as np
from shapely.geometry import Polygon

from computer_vision.vision import Vision
from computer_vision.vision_params_manager import VisionParamsManager
from utils.camera_utils import find_available_camera


class CameraCaptureThread(threading.Thread):
    """
    Background thread that continuously captures frames from a camera device
    and pushes the latest frame and all vision detections to a queue.Queue(maxsize=1).
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

        # Vision wrapper instance (handles smoothing internally)
        self.vision = Vision(self.params_manager.color_params, 
                             self.params_manager.canny_params, 
                             self.params_manager.poly_params)

        # Store precomputed static obstacles and goal
        self.static_obstacles: list[Polygon] = []
        self.goal = None

        # Thread-safe access to robot pose
        self.robot_pose_lock = threading.Lock()
        self.robot_pose = None  # last smoothed robot pose

    def stop(self):
        self._stop_event.set()

    def _grab_first_valid_frame(self, cap, timeout: float):
        """Grab a valid (non-black) frame within timeout seconds."""
        start = time.time()
        while time.time() - start < timeout and not self._stop_event.is_set():
            ok, frame = cap.read()
            if ok and frame is not None and frame.sum() != 0:
                return frame
            time.sleep(0.05)
        return None

    # -------------------------------
    # Thread run loop
    # -------------------------------
    def run(self):
        _, cap = find_available_camera()

        # Initialize static obstacles and goal from first frame
        self.init_obstacles_and_goal(cap)

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            detection_data, overlay_frame = self.process_frame(frame.copy())

            # Push all detection results into the queue
            self.push_processed_data(overlay_frame, detection_data)

            time.sleep(0.01)

        cap.release()

    # -------------------------------
    # Initialization: static obstacles & goal
    # -------------------------------
    def init_obstacles_and_goal(self, cap):
        first_frame = self._grab_first_valid_frame(cap, self.warmup_timeout)
        if first_frame is not None:
            # Detect goal and obstacles
            self.goal = self.vision.detect_goal(first_frame)
            self.static_obstacles = self.vision.detect_obstacles(first_frame)

    # -------------------------------
    # Process single frame
    # -------------------------------
    def process_frame(self, frame: np.ndarray):
        """
        Process incoming frame: detect robot (smoothed internally), overlay obstacles and goal.
        Returns a dict with all detection results and a debug overlay image.
        """
        overlay_frame = frame.copy()

        # Vision wrapper handles smoothing internally
        robot = self.vision.detect_robot(frame)
        goal = self.goal
        obstacles = self.static_obstacles

        with self.robot_pose_lock:
            self.robot_pose = robot

        # Overlay static obstacles
        for poly in obstacles:
            pts = np.array(poly.exterior.coords, np.int32)
            cv2.polylines(overlay_frame, [pts], True, (0, 255, 0), 2)

        # Overlay goal
        if goal and goal["found"] and goal["center"] is not None:
            cv2.circle(overlay_frame, goal["center"], 5, (0, 255, 255), -1)

        # Overlay robot
        if robot and robot["found"]:
            center = robot["center"]
            theta = robot["theta"]
            arrow_length = 60
            dx = int(arrow_length * np.cos(theta))
            dy = int(arrow_length * np.sin(theta))
            cv2.circle(overlay_frame, center, 5, (0, 0, 255), -1)
            cv2.arrowedLine(overlay_frame, center, (center[0]+dx, center[1]+dy), (255, 255, 0), 2, tipLength=0.7)

        # Aggregate all detection data
        detection_data = {
            "robot": robot,
            "goal": goal,
            "obstacles": obstacles
        }

        return detection_data, overlay_frame

    # -------------------------------
    # Queue push
    # -------------------------------
    def push_processed_data(self, frame: np.ndarray, detection_data: dict):
        try:
            self.queue.put_nowait({"frame": frame, "detections": detection_data})
        except queue.Full:
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait({"frame": frame, "detections": detection_data})
            except queue.Full:
                pass

    # -------------------------------
    # Thread-safe access to robot pose
    # -------------------------------
    def get_robot_pose(self):
        with self.robot_pose_lock:
            return self.robot_pose.copy() if self.robot_pose else None
