import numpy as np
from shapely.geometry import Polygon
from computer_vision.computer_vision import ComputerVisionCore as CVCore


class Vision:
    """
    High-level vision wrapper around ComputerVisionCore.
    Stores CV parameters and provides robot/goal/obstacle detection with optional enhancements.
    """

    def __init__(self, params_color, params_canny, params_poly, smoothing_alpha=0.5):
        """
        Args:
            params_color: dict, color thresholds (R/G/B) for CVCore.
            params_canny: dict, Canny edge detection thresholds.
            params_poly: dict, polygon filtering parameters.
            smoothing_alpha: float, optional smoothing factor [0,1] for robot/goal positions.
        """
        self.params_color = params_color
        self.params_canny = params_canny
        self.params_poly = params_poly
        self.smoothing_alpha = smoothing_alpha

        # Caches
        self._last_robot_pose = None
        self._last_goal_pose = None
        self._last_obstacles = None

    # -------------------------
    # Robot Detection
    # -------------------------
    def detect_robot(self, frame, use_cache=True):
        """
        Detect robot in frame. Optionally smooth position using last detection.
        """
        robot_pose = CVCore.detect_robot(frame, self.params_color)

        if robot_pose["found"] and self._last_robot_pose is not None:
            # Simple exponential smoothing for robot center
            cx, cy = robot_pose["center"]
            lx, ly = self._last_robot_pose["center"]
            smoothed_center = (
                int(self.smoothing_alpha * cx + (1 - self.smoothing_alpha) * lx),
                int(self.smoothing_alpha * cy + (1 - self.smoothing_alpha) * ly)
            )
            robot_pose["center"] = smoothed_center
            # Theta smoothing could be added similarly
            # robot_pose["theta"] = self.smoothing_alpha * robot_pose["theta"] + (1 - self.smoothing_alpha) * self._last_robot_pose["theta"]

        if robot_pose["found"]:
            self._last_robot_pose = robot_pose.copy()

        return robot_pose

    # -------------------------
    # Goal Detection
    # -------------------------
    def detect_goal(self, frame, use_cache=True):
        """
        Detect goal in frame. Optionally smooth position using last detection.
        """
        goal_pose = CVCore.detect_goal(frame, self.params_color)

        if goal_pose["found"] and self._last_goal_pose is not None:
            # Simple exponential smoothing for goal center
            cx, cy = goal_pose["center"]
            lx, ly = self._last_goal_pose["center"]
            smoothed_center = (
                int(self.smoothing_alpha * cx + (1 - self.smoothing_alpha) * lx),
                int(self.smoothing_alpha * cy + (1 - self.smoothing_alpha) * ly)
            )
            goal_pose["center"] = smoothed_center

        if goal_pose["found"]:
            self._last_goal_pose = goal_pose.copy()

        return goal_pose

    # -------------------------
    # Obstacle Detection
    # -------------------------
    def detect_obstacles(self, frame, use_cache=True):
        """
        Detect obstacles in frame.
        Returns a list of Shapely Polygon objects.
        """
        obstacles = CVCore.detect_obstacles(frame, self.params_color, self.params_canny, self.params_poly)
        if use_cache:
            self._last_obstacles = obstacles
        return obstacles

    # -------------------------
    # Utility Functions
    # -------------------------
    @staticmethod
    def mm_to_px(mm):
        """Convert millimeters to pixels using CVCore scale."""
        return CVCore.mm_to_px(mm, CVCore.MM_PER_PIXEL)

    @staticmethod
    def px_to_mm(px):
        """Convert pixels to millimeters using CVCore scale."""
        return CVCore.px_to_mm(px, CVCore.MM_PER_PIXEL)

    @staticmethod
    def compute_mm_per_pixel(pt1_px, pt2_px, real_distance_mm):
        """Compute millimeter-per-pixel scale using two known points."""
        return CVCore.compute_mm_per_pixel(pt1_px, pt2_px, real_distance_mm)

    # -------------------------
    # Optional: Get cached data
    # -------------------------
    @property
    def last_robot_pose(self):
        return self._last_robot_pose

    @property
    def last_goal_pose(self):
        return self._last_goal_pose

    @property
    def last_obstacles(self):
        return self._last_obstacles
