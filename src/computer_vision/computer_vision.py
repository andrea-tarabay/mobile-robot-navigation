import cv2
import numpy as np
from typing import Optional
from shapely.geometry import Polygon, box


# -----------------------------
# UTILS (unchanged)
# -----------------------------

def hsv_mask_circular(hsv, hmin, hmax, smin=40, vmin=40):
    if hmin <= hmax:
        return cv2.inRange(hsv, (hmin, smin, vmin), (hmax, 255, 255))
    m1 = cv2.inRange(hsv, (hmin, smin, vmin), (179, 255, 255))
    m2 = cv2.inRange(hsv, (0, smin, vmin), (hmax, 255, 255))
    return cv2.bitwise_or(m1, m2)

def circularity_of_contour(c):
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))

def find_circle_candidates(mask, min_area, max_area, min_circularity=0.6):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        circ = circularity_of_contour(c)
        if circ < min_circularity:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        out.append({"center": (cx, cy), "area": float(a), "circ": float(circ), "contour": c})
    return out

def robot_pair_ok(red, green, max_r_rel_diff=0.25, dist_factor=3.0):
    Ar, Ag = red["area"], green["area"]
    if Ar <= 0 or Ag <= 0:
        return False
    rr = np.sqrt(Ar / np.pi)
    rg = np.sqrt(Ag / np.pi)
    rel = abs(rr - rg) / max(rr, rg)
    if rel > max_r_rel_diff:
        return False
    dx = green["center"][0] - red["center"][0]
    dy = green["center"][1] - red["center"][1]
    dist = np.sqrt(dx*dx + dy*dy)
    if dist > np.sqrt(min(Ar, Ag)) * dist_factor:
        return False
    return True

def score_pair(red, green):
    Ar, Ag = red["area"], green["area"]
    rr = np.sqrt(Ar / np.pi)
    rg = np.sqrt(Ag / np.pi)
    rel_diff = abs(rr - rg) / max(rr, rg)
    dx = green["center"][0] - red["center"][0]
    dy = green["center"][1] - red["center"][1]
    dist = np.sqrt(dx*dx + dy*dy)
    return float(red["circ"] + green["circ"] - 0.0015 * dist - 1.5 * rel_diff)

# =========================================================
# ComputerVisionCore: stateless processing
# =========================================================

class ComputerVisionCore:
    """
    Stateless CV core: processes a frame given parameters.
    """
    # ---- Constants ----
    MM_PER_PIXEL = 0.85

    MARKER_RADIUS_MM = 20
    ROBOT_RADIUS_MM = 72
    GOAL_RADIUS_MM = 50


    @staticmethod
    def detect_robot(frame: np.ndarray, params_color: dict):
        """
        Detect robot using red & green circular markers.

        Returns dict:
            found, center, theta,
            red_center, green_center,
            robot_mask,
            smooth_center, smooth_theta
        """
        overlay = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rough masks
        rough_red_mask = hsv_mask_circular(hsv, params_color["R_hmin"], params_color["R_hmax"], 
                                           params_color["S_min"], params_color["V_min"])
        rough_green_mask = hsv_mask_circular(hsv, params_color["G_hmin"], params_color["G_hmax"], 
                                             params_color["S_min"], params_color["V_min"])
        overlay[rough_red_mask > 0] = (0, 0, 255)
        overlay[rough_green_mask > 0] = (0, 255, 0)
        combined_mask = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # Find candidate circles first
        min_area_marker = np.pi * (ComputerVisionCore.mm_to_px(ComputerVisionCore.MARKER_RADIUS_MM) * 0.5) **2
        max_area_marker = np.pi * (ComputerVisionCore.mm_to_px(ComputerVisionCore.MARKER_RADIUS_MM) * 1.5) **2
        reds = find_circle_candidates(rough_red_mask, min_area=min_area_marker, max_area=max_area_marker)
        greens = find_circle_candidates(rough_green_mask, min_area=min_area_marker, max_area=max_area_marker)

        def _return_not_found():
            return {
                "found": False,
                "center": None,
                "theta": None,
                "red_center": None,
                "red_area": None,
                "green_center": None,
                "green_area": None
            }
            
        if not reds or not greens:
            return _return_not_found()

        best_pair = None
        best_score = -1e12
        for br in reds:
            for bg in greens:
                if not robot_pair_ok(br, bg):
                    continue
                s = score_pair(br, bg)
                if s > best_score:
                    best_score = s
                    best_pair = (br, bg)

        if best_pair is None:
            return _return_not_found()

        br, bg = best_pair
        robot_center = (int((br["center"][0]+bg["center"][0])/2), int((br["center"][1]+bg["center"][1])/2))
        dx, dy = bg["center"][0]-br["center"][0], bg["center"][1]-br["center"][1]
        theta = float(np.arctan2(dy, dx)-np.pi/2)

        return {
            "found": True,
            "center": robot_center,
            "theta": theta,
            "red_center": br["center"],
            "red_area": br["area"],
            "green_center": bg["center"],
            "green_area": bg["area"],
            "robot_mask": combined_mask
        }
    
    @staticmethod
    def detect_goal(frame: np.ndarray, params_color: dict):
        """
        Detect goal using blue circular marker.

        Returns dict:
            found, 
            center,
            goal_mask
        """
        overlay = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rough mask
        rough_blue_mask = hsv_mask_circular(hsv, params_color["B_hmin"], params_color["B_hmax"], 
                                           params_color["S_min"], params_color["V_min"])
        overlay[rough_blue_mask > 0] = (255, 0, 0)
        goal_mask = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # Find candidate circles first
        min_area_goal = np.pi * (ComputerVisionCore.mm_to_px(ComputerVisionCore.GOAL_RADIUS_MM) * 0.5) **2
        max_area_goal = np.pi * (ComputerVisionCore.mm_to_px(ComputerVisionCore.GOAL_RADIUS_MM) * 1.5) **2
        blues = find_circle_candidates(rough_blue_mask, min_area=min_area_goal, max_area=max_area_goal)

        def _return_not_found():
            return {
                "found": False,
                "center": None,
                "goal_mask": None
            }
            
        if not blues:
            return _return_not_found()

        gb = blues[0]
        goal_center = gb["center"]

        return {
            "found": True,
            "center": goal_center,
            "goal_mask": goal_mask
        }
    
    @staticmethod
    def detect_obstacles(frame: np.ndarray, params_color: dict, params_canny: dict, params_poly: dict):
        """
        Detect polygons in the frame using Canny and contour approximation.
        Removes polygons containing the robot position.

        Returns:
            List of polygons (each polygon is a Shapely Polygon)
        """
        frame_copy = frame.copy()
        robot_pose = ComputerVisionCore.detect_robot(frame_copy, params_color)

        # Create robot rectangle if robot is found
        robot_bbox = None
        if robot_pose["found"]:
            # Get robot center
            center = robot_pose["center"]

            x_min = center[0] - ComputerVisionCore.mm_to_px(ComputerVisionCore.ROBOT_RADIUS_MM)
            x_max = center[0] + ComputerVisionCore.mm_to_px(ComputerVisionCore.ROBOT_RADIUS_MM)
            y_min = center[1] - ComputerVisionCore.mm_to_px(ComputerVisionCore.ROBOT_RADIUS_MM)
            y_max = center[1] + ComputerVisionCore.mm_to_px(ComputerVisionCore.ROBOT_RADIUS_MM)

            
            robot_bbox = box(x_min, y_min, x_max, y_max)

        frame_copy = frame.copy()
        goal_pose = ComputerVisionCore.detect_goal(frame_copy, params_color)

        # Create goal rectangle if goal is found
        goal_bbox = None
        if goal_pose["found"]:
            # Get goal center
            center = goal_pose["center"]

            x_min = center[0] - ComputerVisionCore.mm_to_px(ComputerVisionCore.GOAL_RADIUS_MM)
            x_max = center[0] + ComputerVisionCore.mm_to_px(ComputerVisionCore.GOAL_RADIUS_MM)
            y_min = center[1] - ComputerVisionCore.mm_to_px(ComputerVisionCore.GOAL_RADIUS_MM)
            y_max = center[1] + ComputerVisionCore.mm_to_px(ComputerVisionCore.GOAL_RADIUS_MM)

            
            goal_bbox = box(x_min, y_min, x_max, y_max)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        edges = cv2.Canny(blur, params_canny["low"], params_canny["high"])
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, 
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < params_poly["min_area"]:
                continue
            hull = cv2.convexHull(c)
            if len(hull) >= 3:  # at least a triangle
                poly = Polygon([(pt[0][0], pt[0][1]) for pt in hull])
                # Skip polygon if it intersects with robot bounding box
                if robot_bbox is not None and poly.intersects(robot_bbox):
                    continue
                # Skip polygon if it intersects with goal bounding box
                if goal_bbox is not None and poly.intersects(goal_bbox):
                    continue
                obstacles.append(poly)
        return obstacles
    
    @staticmethod
    def init_canny(frame, sigma=0.33):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        v = np.median(gray)

        low = int(max(0, (1.0 - sigma) * v))
        high = int(min(255, (1.0 + sigma) * v))

        return low, high
    
    @staticmethod
    def px_to_mm(distance_px: int, mm_per_pixel: Optional[float] = None):
        """
        Convert distance in pixels to millimeters using known millimeter-per-pixel ratio.
        """
        if mm_per_pixel is None:
            mm_per_pixel = ComputerVisionCore.MM_PER_PIXEL
        return np.asarray(distance_px, dtype=float) * mm_per_pixel
    
    @staticmethod
    def mm_to_px(distance_mm: float, mm_per_pixel: Optional[float] = None):
        """
        Convert distance in millimeters to pixels using known millimeter-per-pixel ratio.
        """
        if mm_per_pixel is None:
            mm_per_pixel = ComputerVisionCore.MM_PER_PIXEL
        return np.round(np.asarray(distance_mm, dtype=float) / mm_per_pixel).astype(int)
    
    @staticmethod
    def compute_mm_per_pixel(point_px_1, point_px_2, real_dist_mm):
        """Compute millimeter-per-pixel scale using two known markers."""
        pixel_dist = np.linalg.norm(np.array(point_px_1, float) - np.array(point_px_2, float))
        return real_dist_mm / pixel_dist
# ===========================================