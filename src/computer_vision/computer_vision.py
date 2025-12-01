from typing import Dict, Any
import cv2
import numpy as np

from src.computer_vision.vision_params_manager import VisionParamsManager

# =========================================================
# UTILS
# =========================================================

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
# COMPUTER VISION CLASS
# =========================================================

class ComputerVision:
    def __init__(self, params: VisionParamsManager):
        self.params = params
        self.static_polys = None
        self.pose_alpha = 0.4
        self.smooth_pose = {"center": None, "theta": None}
        self.robot_state = {"found": False, "center": None, "theta": None,
                            "red_center": None, "green_center": None, "robot_mask": None}
        self.pose_history = []

    # -------------------------------
    def detect_robot(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cp = self.params.color_params
        mask_r = hsv_mask_circular(hsv, cp["R_hmin"], cp["R_hmax"])
        mask_g = hsv_mask_circular(hsv, cp["G_hmin"], cp["G_hmax"])
        reds = find_circle_candidates(mask_r, self.params.robot_area["min"], self.params.robot_area["max"])
        greens = find_circle_candidates(mask_g, self.params.robot_area["min"], self.params.robot_area["max"])

        best = None
        best_score = -1e9
        for r in reds:
            for g in greens:
                if not robot_pair_ok(r, g):
                    continue
                s = score_pair(r, g)
                if s > best_score:
                    best_score = s
                    best = (r, g)

        if best is None:
            self.robot_state.update({"found": False, "center": None, "theta": None,
                                     "red_center": None, "green_center": None, "robot_mask": None,
                                     "smooth_center": self.smooth_pose["center"],
                                     "smooth_theta": self.smooth_pose["theta"]})
            return self.robot_state

        red, green = best
        rc, gc = red["center"], green["center"]
        center = (int((rc[0]+gc[0])/2), int((rc[1]+gc[1])/2))
        dx, dy = gc[0]-rc[0], gc[1]-rc[1]
        theta = float(np.arctan2(dy, dx)-np.pi/2)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [red["contour"]], -1, 255, -1)
        cv2.drawContours(mask, [green["contour"]], -1, 255, -1)

        if self.smooth_pose["center"] is None:
            self.smooth_pose["center"] = center
            self.smooth_pose["theta"] = theta
        else:
            a = self.pose_alpha
            px, py = self.smooth_pose["center"]
            sx = (1-a)*px + a*center[0]
            sy = (1-a)*py + a*center[1]
            self.smooth_pose["center"] = (int(sx), int(sy))
            pt = self.smooth_pose["theta"]
            st = np.arctan2((1-a)*np.sin(pt)+a*np.sin(theta), (1-a)*np.cos(pt)+a*np.cos(theta))
            self.smooth_pose["theta"] = float(st)

        self.robot_state.update({"found": True, "center": center, "theta": theta,
                                 "red_center": rc, "green_center": gc, "robot_mask": mask,
                                 "smooth_center": self.smooth_pose["center"],
                                 "smooth_theta": self.smooth_pose["theta"]})
        return self.robot_state

    # -------------------------------
    def blur_robot(self, frame, state):
        if state is None or not state.get("found") or state.get("robot_mask") is None:
            return frame.copy()
        mask = state["robot_mask"].copy()
        mask_area = cv2.countNonZero(mask)
        est_radius = int(max(15, 3.0*np.sqrt(mask_area/np.pi)))
        ksize = int(0.5*est_radius)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.dilate(mask, k, iterations=1)
        cleaned = cv2.inpaint(frame, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        return cleaned

    # -------------------------------
    def process(self, frame, skip_blur=False, log_pose=True):
        st = self.detect_robot(frame)
        frame_clean = frame if skip_blur else self.blur_robot(frame, st)

        if self.static_polys is not None:
            polys = list(self.static_polys)
        else:
            gray = cv2.cvtColor(frame_clean, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, self.params.canny_params["low"], self.params.canny_params["high"])
            edges = strengthen_edges(edges, ksize=3, iterations=3)
            edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=2)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = []
            for c in contours:
                a = cv2.contourArea(c)
                if not (self.params.poly_params["min_area"] < a < self.params.poly_params["max_area"]):
                    continue
                epsilon = 0.016*cv2.arcLength(c, True)
                poly = cv2.approxPolyDP(c, epsilon, True)
                pts = [(int(p[0][0]), int(p[0][1])) for p in poly]
                polys.append(pts)

        if log_pose:
            self._record_pose(st)
        return polys, st

    # -------------------------------
    def _record_pose(self, st):
        if st is None or not st.get("found"):
            return
        center = st.get("smooth_center") or st.get("center")
        theta = st.get("smooth_theta") or st.get("theta")
        if center is None or theta is None:
            return
        self.pose_history.append((float(center[0]), float(center[1]), float(theta)))

    # -------------------------------
    def get_pose_array(self, as_numpy=True, clear=False):
        if as_numpy:
            data = np.array(self.pose_history, dtype=np.float32)
            if data.size==0:
                data = data.reshape(0,3)
        else:
            data = self.pose_history
        if clear:
            self.pose_history=[]
        return data

    def reset_pose_history(self):
        self.pose_history=[]

    # -----------------------------------------------------
    def auto_init_colors(self, frame, expand_h=5, expand_sv=30):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rough masks
        rough_red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)),
            cv2.inRange(hsv, (160, 50, 50), (179, 255, 255))
        )
        rough_green_mask = cv2.inRange(hsv, (40, 50, 50), (90, 255, 255))

        red_candidates = find_circle_candidates(rough_red_mask, 10, 10000)
        green_candidates = find_circle_candidates(rough_green_mask, 10, 10000)
        if not red_candidates or not green_candidates:
            raise RuntimeError("Robot markers not detected for auto calibration.")

        def compute_hsv_range(mask):
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                return 0, 179, 0, 255, 0, 255
            pixels = hsv[ys, xs]
            hmin = max(0, int(np.min(pixels[:, 0])) - expand_h)
            hmax = min(179, int(np.max(pixels[:, 0])) + expand_h)
            return hmin, hmax

        # Red HSV
        red_mask_combined = np.zeros(frame.shape[:2], dtype=np.uint8)
        for c in red_candidates:
            cv2.drawContours(red_mask_combined, [c["contour"]], -1, 255, -1)
        R_hmin, R_hmax = compute_hsv_range(red_mask_combined)

        # Green HSV
        green_mask_combined = np.zeros(frame.shape[:2], dtype=np.uint8)
        for c in green_candidates:
            cv2.drawContours(green_mask_combined, [c["contour"]], -1, 255, -1)
        G_hmin, G_hmax = compute_hsv_range(green_mask_combined)

        # Robot area
        all_areas = [c["area"] for c in red_candidates + green_candidates]
        a_min = max(5, int(min(all_areas) * 0.8))
        a_max = int(max(all_areas) * 1.2)

        print(f"[AUTO INIT] Red H: {R_hmin}-{R_hmax}, Green H: {G_hmin}-{G_hmax}, Area: {a_min}-{a_max}")

        return {
            "color_params": {"R_hmin": R_hmin, "R_hmax": R_hmax, "G_hmin": G_hmin, "G_hmax": G_hmax},
            "robot_area": {"min": a_min, "max": a_max}
        }
    
    def init_canny(self, frame, sigma=0.33):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        v = np.median(gray)

        low = int(max(0, (1.0 - sigma) * v))
        high = int(min(255, (1.0 + sigma) * v))

        print(f"[AUTO INIT] Canny thresholds: low={low}, high={high}")

        return low, high
# =========================================================