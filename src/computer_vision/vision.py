import cv2
import numpy as np
import os

# ============================================
# UTILS
# ============================================

def hsv_mask_circular(hsv, hmin, hmax, smin=40, vmin=40):
    """
    Masque HSV en tenant compte du wrap 179 -> 0.
    hmin, hmax dans [0,179].
    """
    if hmin <= hmax:
        lower = np.array([hmin, smin, vmin])
        upper = np.array([hmax, 255, 255])
        return cv2.inRange(hsv, lower, upper)

    lower1 = np.array([hmin, smin, vmin])
    upper1 = np.array([179, 255, 255])
    lower2 = np.array([0,   smin, vmin])
    upper2 = np.array([hmax, 255, 255])

    return cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2)
    )


def circularity_of_contour(c):
    """ 4πA / P² """
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def find_circle_candidates(mask, min_area, max_area, min_circularity=0.7):
    """
    Retourne une liste de candidats ronds:
    [{"center":(x,y), "area":A, "circ":C, "contour":c}]
    On préfère être strict -> si circ trop faible ou aire hors range, rejet.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        circ = circularity_of_contour(c)
        if circ < min_circularity:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        candidates.append({
            "center": (cx, cy),
            "area": float(area),
            "circ": float(circ),
            "contour": c
        })

    return candidates


def robot_pair_ok(red, green, max_r_rel_diff=0.15, dist_factor=1.0):
    """
    Conditions strictes:
    - rayons égaux ±15%
    - distance centres <= sqrt(min(area)) * dist_factor
    """
    if red is None or green is None:
        return False

    Ar = red["area"]
    Ag = green["area"]
    if Ar <= 0 or Ag <= 0:
        return False

    rr = np.sqrt(Ar / np.pi)
    rg = np.sqrt(Ag / np.pi)
    max_r = max(rr, rg)
    if max_r <= 1e-6:
        return False

    rel_diff = abs(rr - rg) / max_r
    if rel_diff > max_r_rel_diff:
        return False

    dx = green["center"][0] - red["center"][0]
    dy = green["center"][1] - red["center"][1]
    dist = np.sqrt(dx*dx + dy*dy)

    max_dist = np.sqrt(min(Ar, Ag)) * dist_factor
    if dist > max_dist:
        return False

    return True


def score_pair(red, green):
    """
    Score de matching (plus grand = meilleur).
    On favorise:
    - circularité haute
    - aires proches
    - distance raisonnable
    """
    Ar, Ag = red["area"], green["area"]
    rr = np.sqrt(Ar / np.pi)
    rg = np.sqrt(Ag / np.pi)

    dx = green["center"][0] - red["center"][0]
    dy = green["center"][1] - red["center"][1]
    dist = np.sqrt(dx*dx + dy*dy)

    # écart relatif rayon
    rel_diff = abs(rr - rg) / max(rr, rg)

    # score: circ + circ - pénalité distance - pénalité rel_diff
    score = (red["circ"] + green["circ"]) - 0.002*dist - 2.0*rel_diff
    return float(score)


# ============================================
# VISION CLASS
# ============================================

class Vision:
    def __init__(self):
        self.color_params = None   # HSV thresholds + aires
        self.canny_params = None   # Canny thresholds + aires

        self.robot_state = {
            "found": False,
            "center": None,
            "red_center": None,
            "green_center": None,
            "theta": None,
            "red_area": 0.0,
            "green_area": 0.0,
            "robot_mask": None  # masque final du robot (uint8)
        }
        #self.debug_remove_robot = 0   
        self.debug_remove_robot = 1 # pour debug robot removal

        BASE = os.path.dirname(os.path.abspath(__file__))
        self.IMGS = os.path.join(BASE, "images")

    def load(self, filename):
        path = os.path.join(self.IMGS, filename)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Impossible de charger {path}")
        return img

    # =======================================
    #  DETECTION ROBOT (STRICTE, OPTIMALE)
    # =======================================
    def detect_robot(self, frame):
        """
        Détecte toutes les bulles rouges/vertes, cherche la meilleure paire
        respectant TES conditions.
        Si aucune paire valide -> found=False (préférence faux négatif).
        """
        if self.color_params is None:
            self.robot_state.update({
                "found": False, "center": None,
                "red_center": None, "green_center": None,
                "theta": None, "red_area": 0.0, "green_area": 0.0,
                "robot_mask": None
            })
            return self.robot_state

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        R_hmin = self.color_params["R_hmin"]
        R_hmax = self.color_params["R_hmax"]
        G_hmin = self.color_params["G_hmin"]
        G_hmax = self.color_params["G_hmax"]
        minA = self.color_params["min_area"]
        maxA = self.color_params["max_area"]

        mask_red = hsv_mask_circular(hsv, R_hmin, R_hmax)
        mask_green = hsv_mask_circular(hsv, G_hmin, G_hmax)

        # candidats stricts
        reds = find_circle_candidates(mask_red, minA, maxA, min_circularity=0.6)
        greens = find_circle_candidates(mask_green, minA, maxA, min_circularity=0.6)

        best_pair = None
        best_score = -1e9

        for r in reds:
            for g in greens:
                if not robot_pair_ok(r, g, max_r_rel_diff=0.15, dist_factor=2.4): #distance centres <= sqrt(min(area)) * dist_factor
                    continue
                s = score_pair(r, g)
                if s > best_score:
                    best_score = s
                    best_pair = (r, g)

        if best_pair is None:
            self.robot_state.update({
                "found": False, "center": None,
                "red_center": None, "green_center": None,
                "theta": None, "red_area": 0.0, "green_area": 0.0,
                "robot_mask": None
            })
            return self.robot_state

        red, green = best_pair
        red_center = red["center"] 
        green_center = green["center"]

        center = (
            int((red_center[0] + green_center[0]) / 2),
            int((red_center[1] + green_center[1]) / 2)
        )

        dx = green_center[0] - red_center[0]
        dy = green_center[1] - red_center[1]
        theta = float(np.arctan2(dy, dx) - np.pi/2 )  # orientation robot
        print("Theta (deg):", np.degrees(theta))

        # masque robot = union des 2 contours, dilaté
        robot_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(robot_mask, [red["contour"]], -1, 255, -1)
        cv2.drawContours(robot_mask, [green["contour"]], -1, 255, -1)

        # dilatation pour couvrir tout le robot
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        robot_mask = cv2.dilate(robot_mask, k, iterations=1)

        self.robot_state.update({
            "found": True,
            "center": center,
            "red_center": red_center,
            "green_center": green_center,
            "theta": theta,
            "red_area": red["area"],
            "green_area": green["area"],
            "robot_mask": robot_mask
        })
        return self.robot_state

    # =======================================
    #  INITIALISATION COULEUR (ROBOT)
    # =======================================
    def init_colors(self, frame):
        cv2.namedWindow("COLOR INIT")

        R_hmin, R_hmax = 170, 10
        G_hmin, G_hmax = 40, 90
        minA, maxA = 50, 2000

        cv2.createTrackbar("Red Hmin",   "COLOR INIT", R_hmin, 179, lambda x: None)
        cv2.createTrackbar("Red Hmax",   "COLOR INIT", R_hmax, 179, lambda x: None)
        cv2.createTrackbar("Green Hmin", "COLOR INIT", G_hmin, 179, lambda x: None)
        cv2.createTrackbar("Green Hmax", "COLOR INIT", G_hmax, 179, lambda x: None)
        cv2.createTrackbar("Min Area",   "COLOR INIT", minA, 5000, lambda x: None)
        cv2.createTrackbar("Max Area",   "COLOR INIT", maxA, 20000, lambda x: None)

        while True:
            R_hmin = cv2.getTrackbarPos("Red Hmin",   "COLOR INIT")
            R_hmax = cv2.getTrackbarPos("Red Hmax",   "COLOR INIT")
            G_hmin = cv2.getTrackbarPos("Green Hmin", "COLOR INIT")
            G_hmax = cv2.getTrackbarPos("Green Hmax", "COLOR INIT")
            minA   = cv2.getTrackbarPos("Min Area",   "COLOR INIT")
            maxA   = cv2.getTrackbarPos("Max Area",   "COLOR INIT")

            self.color_params = {
                "R_hmin": R_hmin, "R_hmax": R_hmax,
                "G_hmin": G_hmin, "G_hmax": G_hmax,
                "min_area": minA, "max_area": maxA
            }

            state = self.detect_robot(frame)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_red = hsv_mask_circular(hsv, R_hmin, R_hmax)
            mask_green = hsv_mask_circular(hsv, G_hmin, G_hmax)

            debug = frame.copy()
            overlay = debug.copy()
            overlay[mask_red > 0] = (0, 0, 255)
            overlay[mask_green > 0] = (0, 255, 0)
            debug = cv2.addWeighted(overlay, 0.3, debug, 0.7, 0)

            # dessiner tous les candidats (petits cercles)
            reds = find_circle_candidates(mask_red, minA, maxA, 0.7)
            greens = find_circle_candidates(mask_green, minA, maxA, 0.7)
            for r in reds:
                cv2.circle(debug, r["center"], 6, (0, 0, 200), 2)
            for g in greens:
                cv2.circle(debug, g["center"], 6, (0, 200, 0), 2)

            if state["red_center"] is not None:
                cv2.circle(debug, state["red_center"], 10, (0, 0, 255), -1)
                cv2.putText(debug, f"{int(state['red_area'])} px",
                            (state["red_center"][0]-20, state["red_center"][1]-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            if state["green_center"] is not None:
                cv2.circle(debug, state["green_center"], 10, (0, 255, 0), -1)
                cv2.putText(debug, f"{int(state['green_area'])} px",
                            (state["green_center"][0]-20, state["green_center"][1]-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            if state["found"]:
                cx, cy = state["center"]
                cv2.circle(debug, (cx, cy), 6, (255,255,255), -1)
                length = 50
                th = state["theta"]
                x2 = int(cx + length*np.cos(th))
                y2 = int(cy + length*np.sin(th))
                cv2.arrowedLine(debug, (cx, cy), (x2, y2),
                                (255,255,255), 2, tipLength=0.3)

                cv2.putText(debug, "Robot: FOUND (flag=1)", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
            else:
                cv2.putText(debug, "Robot: NOT FOUND (flag=0)", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

            cv2.imshow("COLOR INIT", debug)

            k = cv2.waitKey(1) & 0xFF
            if k in [13, ord(' ')]:  # ENTER/espace
                break
            if k in [27, ord('q')]:
                break

        cv2.destroyWindow("COLOR INIT")
        # color_params déjà stockés

    # =======================================
    #  SUPPRESSION ROBOT (INPAINT FORT)
    # =======================================
    def remove_robot(self, frame):
        """
        1) détecte robot STRICT
        2) inpaint sur un masque dilaté
        => robot invisible pour Canny.
        
        Si self.debug_remove_robot == 1 :
            affiche l'image nettoyée (pour la documentation)
        """
        state = self.detect_robot(frame)
        if not state["found"] or state["robot_mask"] is None:
            cleaned = frame.copy()

            # --- DEBUG OPTIONNEL ---
            if getattr(self, "debug_remove_robot", 0) == 1:
                cv2.imshow("REMOVE_ROBOT_DEBUG", cleaned)
                cv2.waitKey(1)

            return cleaned, None

        mask = state["robot_mask"]

        # inpaint
        removed = cv2.inpaint(frame, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

        # --- DEBUG OPTIONNEL ---
        if getattr(self, "debug_remove_robot", 0) == 1:
            cv2.imshow("REMOVE_ROBOT_DEBUG", removed)
            cv2.waitKey(1)

        return removed, mask
    # =======================================
    #  PREPROCESS COMMUN CANNY
    # =======================================
    def _preprocess_for_canny(self, frame):
        clean, robot_mask = self.remove_robot(frame)

        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)

        # même filtre partout
        smooth = cv2.bilateralFilter(eq, d=9, sigmaColor=50, sigmaSpace=50)
        return clean, smooth, robot_mask

    # =======================================
    #  INITIALISATION CANNY (identique final)
    # =======================================
    def init_canny(self, frame):
        cv2.namedWindow("CANNY INIT")

        low, high = 30, 120
        minA, maxA = 300, 20000

        cv2.createTrackbar("Canny Low",  "CANNY INIT", low, 255, lambda x: None)
        cv2.createTrackbar("Canny High", "CANNY INIT", high, 255, lambda x: None)
        cv2.createTrackbar("Min Area",   "CANNY INIT", minA, 50000, lambda x: None)
        cv2.createTrackbar("Max Area",   "CANNY INIT", maxA, 80000, lambda x: None)

        while True:
            low  = cv2.getTrackbarPos("Canny Low",  "CANNY INIT")
            high = cv2.getTrackbarPos("Canny High", "CANNY INIT")
            minA = cv2.getTrackbarPos("Min Area",   "CANNY INIT")
            maxA = cv2.getTrackbarPos("Max Area",   "CANNY INIT")

            clean, smooth, robot_mask = self._preprocess_for_canny(frame)

            edges = cv2.Canny(smooth, low, high)

            # affichage basé sur l'image CANNNY (comme tu veux)
            debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                a = cv2.contourArea(c)
                if minA < a < maxA:
                    poly = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
                    cv2.polylines(debug, [poly], True, (0,255,0), 2)

            cv2.imshow("CANNY INIT", debug)

            k = cv2.waitKey(1) & 0xFF
            if k in [13, ord(' ')]:
                break
            if k in [27, ord('q')]:
                break

        cv2.destroyWindow("CANNY INIT")

        self.canny_params = {
            "low": low, "high": high,
            "min_area": minA, "max_area": maxA
        }

    # =======================================
    #  DÉTECTION POLYGONES FINAL
    # =======================================
    def process(self, frame):
        if self.canny_params is None:
            raise RuntimeError("Canny non initialisé : appelle init_canny() d'abord.")

        clean, smooth, robot_mask = self._preprocess_for_canny(frame)

        edges = cv2.Canny(
            smooth,
            self.canny_params["low"],
            self.canny_params["high"]
        )

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polys = []
        for c in contours:
            a = cv2.contourArea(c)
            if not (self.canny_params["min_area"] < a < self.canny_params["max_area"]):
                continue

            poly = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
            pts = [(int(p[0][0]), int(p[0][1])) for p in poly]

            # --- suppression post-canny si ça tombe sur le robot ---
            if robot_mask is not None:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    if robot_mask[cy, cx] > 0:
                        continue

            polys.append(pts)

        return polys

    # =======================================
    #  ACCÈS SIMPLE AU POSE ROBOT
    # =======================================
    def get_robot_pose(self, frame):
        """
        Retourne (found, center(x,y), theta_rad).
        Utilise detect_robot().
        """
        st = self.detect_robot(frame)
        return st["found"], st["center"], st["theta"]


# ============================================
# EXEMPLE IMAGE FIXE
# ============================================
if __name__ == "__main__":

    vision = Vision()
    img = vision.load("table7.jpg")

    print("=== INIT COULEURS ===")
    vision.init_colors(img)

    print("=== INIT CANNY ===")
    vision.init_canny(img)

    polys = vision.process(img)
    robot = vision.detect_robot(img)  # robot["center"], robot["theta"], robot["found"]

    print("Robot flag:", int(robot["found"]))
    print("Robot center:", robot["center"])
    print("Robot theta (rad):", robot["theta"])

    out = img.copy()

    for poly in polys:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0,255,0), 3)

    if robot["found"]:
        cx, cy = robot["center"]
        cv2.circle(out, (cx, cy), 7, (255,255,255), -1)
        th = robot["theta"]
        x2 = int(cx + 60*np.cos(th))
        y2 = int(cy + 60*np.sin(th))
        cv2.arrowedLine(out, (cx, cy), (x2, y2), (0,0,255), 3, tipLength=0.3)

    cv2.imshow("FINAL", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()