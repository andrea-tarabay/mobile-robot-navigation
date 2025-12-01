import cv2
import numpy as np
from shapely.geometry import Polygon, Point


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

def strengthen_edges(edges, ksize=3, iterations=1):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=iterations)

# =========================================================
# ComputerVisionCore: stateless processing
# =========================================================

class ComputerVisionCore:
    """
    Stateless CV core: processes a frame given parameters.
    """

    @staticmethod
    def detect_robot(frame: np.ndarray):
        """
        Detect robot using red & green circular markers.

        Returns dict:
            found, center, theta,
            red_center, green_center,
            robot_mask,
            smooth_center, smooth_theta
        """
        # --------- Auto-detect robot colors---------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rough masks
        rough_red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)),
            cv2.inRange(hsv, (160, 50, 50), (179, 255, 255))
        )
        rough_green_mask = cv2.inRange(hsv, (40, 50, 50), (90, 255, 255))

        # Find candidate circles first
        reds = find_circle_candidates(rough_red_mask, min_area=10, max_area=10000)
        greens = find_circle_candidates(rough_green_mask, min_area=10, max_area=10000)
        # ------------------------------

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

        if len(reds) > 1 or len(greens) > 1:
            raise RuntimeError("Multiple robot markers detected during auto calibration.")

        if not reds or not greens:
            return _return_not_found()
        elif reds[0]["area"] > 2000 or greens[0]["area"] > 2000:
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

        # build combined mask
        comb_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(comb_mask, [br["contour"]], -1, 255, -1)
        cv2.drawContours(comb_mask, [bg["contour"]], -1, 255, -1)

        return {
            "found": True,
            "center": robot_center,
            "theta": theta,
            "red_center": br["center"],
            "red_area": br["area"],
            "green_center": bg["center"],
            "green_area": bg["area"],
        }
    
    @staticmethod
    def detect_obstacles(frame: np.ndarray, min_area: int, canny_low=100, canny_high=200):
        """
        Detect polygons in the frame using Canny and contour approximation.
        Removes polygons containing the robot position.

        Returns:
            List of polygons (each polygon is a Shapely Polygon)
        """
        frame_copy = frame.copy()
        robot_pose = ComputerVisionCore.detect_robot(frame_copy)

        # If robot found, create a list of points to exclude
        not_obstacle_points = []
        if robot_pose["found"]:
            not_obstacle_points.append(Point(robot_pose["center"]))
            if robot_pose["red_center"]:
                not_obstacle_points.append(Point(robot_pose["red_center"]))
            if robot_pose["green_center"]:
                not_obstacle_points.append(Point(robot_pose["green_center"]))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        edges = cv2.Canny(blur, canny_low, canny_high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, 
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            hull = cv2.convexHull(c)
            if len(hull) >= 3:  # at least a triangle
                poly = Polygon([(pt[0][0], pt[0][1]) for pt in hull])
                # Skip polygon if it contains any "not_obstacle" points
                if any(poly.contains(p) for p in not_obstacle_points):
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
# =========================================================