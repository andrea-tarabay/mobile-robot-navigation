"""
Pipeline vision simplifiée :
- Etape 1 : init couleurs robot (trackbars rouge/vert) sur l'image brute
- Etape 2 : blur du robot proportionnel à sa taille pour le supprimer des obstacles
- Etape 3 : init Canny (trackbars low/high) avec aperçu edges
- Etape 4 : init polygones (trackbars min/max area) avec aperçu des obstacles
- Etape 5 : détection continue (robot + obstacles) sur image fixe ou webcam

J'ai volontairement retiré toute la logique de warp/crop et le cache de map pour raccourcir
et clarifier : une seule fonction initialize() fait les réglages, ensuite process() utilise
les params figés. Les commentaires décrivent chaque fonction.
"""

import cv2
import numpy as np
import os

# =========================================================
# UTILS
# =========================================================

def hsv_mask_circular(hsv, hmin, hmax, smin=40, vmin=40):
    """Masque HSV en tenant compte du wrap 179->0 (rouge)."""
    if hmin <= hmax:
        return cv2.inRange(hsv, (hmin, smin, vmin), (hmax, 255, 255))
    m1 = cv2.inRange(hsv, (hmin, smin, vmin), (179, 255, 255))
    m2 = cv2.inRange(hsv, (0,    smin, vmin), (hmax, 255, 255))
    return cv2.bitwise_or(m1, m2)


def circularity_of_contour(c):
    """Retourne 4πA / P² pour juger si un contour est rond."""
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def find_circle_candidates(mask, min_area, max_area, min_circularity=0.6):
    """Détecte des bulles rondes dans un masque binaire."""
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
    """Filtre simple : rayons proches et distance raisonnable."""
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
    """Score heuristique pour choisir la meilleure paire rouge/vert."""
    Ar, Ag = red["area"], green["area"]
    rr = np.sqrt(Ar / np.pi)
    rg = np.sqrt(Ag / np.pi)
    rel_diff = abs(rr - rg) / max(rr, rg)
    dx = green["center"][0] - red["center"][0]
    dy = green["center"][1] - red["center"][1]
    dist = np.sqrt(dx*dx + dy*dy)
    return float(red["circ"] + green["circ"] - 0.0015 * dist - 1.5 * rel_diff)


def strengthen_edges(edges, ksize=3, iterations=1):
    """
    Ferme les petits trous pour souder deux traits séparés d'1 px.
    Ajuste ksize/iterations pour plus ou moins de tolérance.
    """
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=iterations)


# =========================================================
# VISION CLASS
# =========================================================

class Vision:
    def __init__(self):
        # seuils couleurs robot (HSV)
        self.color_params = {"R_hmin": 170, "R_hmax": 10, "G_hmin": 40, "G_hmax": 90}
        # seuils Canny
        self.canny_params = {"low": 60, "high": 140}
        # aires obstacles
        self.poly_params = {"min_area": 300, "max_area": 20000}
        # aires robot (pour filtrer les bulles)
        self.robot_area = {"min": 20, "max": 8000}

        # obstacles figés à l'init (pour ne pas recalculer en live)
        self.static_polys = None

        # lissage position/orientation
        self.pose_alpha = 0.4
        self.smooth_pose = {"center": None, "theta": None}

        # état robot basique
        self.robot_state = {
            "found": False,
            "center": None,
            "theta": None,
            "red_center": None,
            "green_center": None,
            "robot_mask": None,
        }

        # historique des poses pour export (tests Kalman, etc.)
        self.pose_history = []

        base = os.path.dirname(os.path.abspath(__file__))
        self.IMGS = os.path.join(base, "images")

    # -----------------------------------------------------
    def load(self, filename):
        """Charge une image depuis src/computer_vision/images/filename."""
        path = os.path.join(self.IMGS, filename)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Impossible de charger {path}")
        return img

    # -----------------------------------------------------
    def detect_robot(self, frame):
        """
        Détecte rouge/vert et retourne l'état robot. Si non trouvé -> found=False.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        R_hmin, R_hmax = self.color_params["R_hmin"], self.color_params["R_hmax"]
        G_hmin, G_hmax = self.color_params["G_hmin"], self.color_params["G_hmax"]

        mask_r = hsv_mask_circular(hsv, R_hmin, R_hmax)
        mask_g = hsv_mask_circular(hsv, G_hmin, G_hmax)

        reds = find_circle_candidates(mask_r, self.robot_area["min"], self.robot_area["max"], 0.6)
        greens = find_circle_candidates(mask_g, self.robot_area["min"], self.robot_area["max"], 0.6)

        best = None
        best_score = -1e9
        for r in reds:
            for g in greens:
                if not robot_pair_ok(r, g, max_r_rel_diff=0.25, dist_factor=3.0):
                    continue
                s = score_pair(r, g)
                if s > best_score:
                    best_score = s
                    best = (r, g)

        if best is None:
            # si je perds le robot, je garde la pose lissée précédente pour info
            self.robot_state.update({
                "found": False,
                "center": None,
                "theta": None,
                "red_center": None,
                "green_center": None,
                "robot_mask": None,
                "smooth_center": self.smooth_pose["center"],
                "smooth_theta": self.smooth_pose["theta"],
            })
            return self.robot_state

        red, green = best
        rc, gc = red["center"], green["center"]
        center = (int((rc[0] + gc[0]) / 2), int((rc[1] + gc[1]) / 2))
        dx, dy = gc[0] - rc[0], gc[1] - rc[1]
        theta = float(np.arctan2(dy, dx) - np.pi / 2.0)  # orientation rouge->vert

        # masque du robot = union des 2 contours dilatés (on s'en sert pour blur)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [red["contour"]], -1, 255, -1)
        cv2.drawContours(mask, [green["contour"]], -1, 255, -1)

        # lissage simple (EMA) pour réduire les sauts
        if self.smooth_pose["center"] is None:
            self.smooth_pose["center"] = center
            self.smooth_pose["theta"] = theta
        else:
            a = self.pose_alpha
            px, py = self.smooth_pose["center"]
            sx = (1 - a) * px + a * center[0]
            sy = (1 - a) * py + a * center[1]
            self.smooth_pose["center"] = (int(sx), int(sy))
            # lissage d'angle via sin/cos
            pt = self.smooth_pose["theta"]
            st = np.arctan2((1 - a) * np.sin(pt) + a * np.sin(theta),
                            (1 - a) * np.cos(pt) + a * np.cos(theta))
            self.smooth_pose["theta"] = float(st)

        self.robot_state.update({
            "found": True,
            "center": center,
            "theta": theta,
            "red_center": rc,
            "green_center": gc,
            "robot_mask": mask,
            "smooth_center": self.smooth_pose["center"],
            "smooth_theta": self.smooth_pose["theta"],
        })
        return self.robot_state

    # -----------------------------------------------------
    def _record_pose(self, st):
        """
        Stocke (x, y, theta) dans l'historique dès que le robot est trouvé.
        Utilise la version lissée si dispo, sinon la mesure brute.
        """
        if st is None or not st.get("found"):
            return
        center = st.get("smooth_center") or st.get("center")
        theta = st.get("smooth_theta")
        if theta is None:
            theta = st.get("theta")
        if center is None or theta is None:
            return
        self.pose_history.append((float(center[0]), float(center[1]), float(theta)))

    # -----------------------------------------------------
    def blur_robot(self, frame, state):
        """
        Floute le robot de façon proportionnelle à sa taille pour qu'il disparaisse des obstacles.
        - on dilate le masque avec un rayon basé sur la plus grande aire détectée
        - on applique un inpaint pour remplir la zone
        """
        if state is None or not state.get("found") or state.get("robot_mask") is None:
            return frame.copy()

        mask = state["robot_mask"].copy()

        # rayon proportionnel à la taille du masque (plus le robot est gros, plus on dilate)
        mask_area = cv2.countNonZero(mask)
        est_radius = int(max(15, 3.0 * np.sqrt(mask_area / np.pi)))  # 3x rayon approx

        #ksize = max(3, 2 * est_radius + 1)
        ksize = int(0.5 * est_radius)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.dilate(mask, k, iterations=1)

        # inpaint pour supprimer totalement le robot
        cleaned = cv2.inpaint(frame, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        return cleaned

    # -----------------------------------------------------
    def init_colors(self, frame):
        """Trackbars pour Hmin/Hmax rouge/vert + aires min/max avec aperçu masques."""
        win = "INIT COLORS"
        cv2.namedWindow(win)
        cv2.createTrackbar("R_hmin", win, self.color_params["R_hmin"], 179, lambda x: None)
        cv2.createTrackbar("R_hmax", win, self.color_params["R_hmax"], 179, lambda x: None)
        cv2.createTrackbar("G_hmin", win, self.color_params["G_hmin"], 179, lambda x: None)
        cv2.createTrackbar("G_hmax", win, self.color_params["G_hmax"], 179, lambda x: None)
        cv2.createTrackbar("A_min",  win, int(self.robot_area["min"]), 5000, lambda x: None)
        cv2.createTrackbar("A_max",  win, int(self.robot_area["max"]), 20000, lambda x: None)

        while True:
            self.color_params["R_hmin"] = cv2.getTrackbarPos("R_hmin", win)
            self.color_params["R_hmax"] = cv2.getTrackbarPos("R_hmax", win)
            self.color_params["G_hmin"] = cv2.getTrackbarPos("G_hmin", win)
            self.color_params["G_hmax"] = cv2.getTrackbarPos("G_hmax", win)
            self.robot_area["min"] = max(5, cv2.getTrackbarPos("A_min", win))
            self.robot_area["max"] = max(self.robot_area["min"] + 10, cv2.getTrackbarPos("A_max", win))

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_r = hsv_mask_circular(hsv, self.color_params["R_hmin"], self.color_params["R_hmax"])
            mask_g = hsv_mask_circular(hsv, self.color_params["G_hmin"], self.color_params["G_hmax"])

            # overlay des masques pour bien voir ce qui est capté
            overlay = frame.copy()
            overlay[mask_r > 0] = (0, 0, 255)
            overlay[mask_g > 0] = (0, 255, 0)
            dbg = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            st = self.detect_robot(frame)
            if st["found"]:
                cv2.circle(dbg, st["red_center"], 6, (0, 0, 255), -1)
                cv2.circle(dbg, st["green_center"], 6, (0, 255, 0), -1)
                cv2.circle(dbg, st["center"], 6, (255, 255, 255), -1)
                th = st["theta"]
                x2 = int(st["center"][0] + 50 * np.cos(th))
                y2 = int(st["center"][1] + 50 * np.sin(th))
                cv2.arrowedLine(dbg, st["center"], (x2, y2), (255, 255, 255), 2)
                cv2.putText(dbg, "FOUND", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(dbg, "NOT FOUND", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # aperçu des aires min/max (disques de rayon équivalent)
            r_min = int(np.sqrt(self.robot_area["min"] / np.pi))
            r_max = int(np.sqrt(self.robot_area["max"] / np.pi))
            cv2.circle(dbg, (40, 80), r_min, (200, 200, 200), 2)
            cv2.putText(dbg, f"Amin={int(self.robot_area['min'])}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            cv2.circle(dbg, (120, 80), r_max, (150, 150, 150), 2)
            cv2.putText(dbg, f"Amax={int(self.robot_area['max'])}", (90, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)

            cv2.imshow(win, dbg)
            k = cv2.waitKey(1) & 0xFF
            if k in [13, ord(' ')]:  # ENTER / espace
                break
            if k in [27, ord('q')]:
                break
        cv2.destroyWindow(win)

    # -----------------------------------------------------
    def init_canny(self, frame_no_robot):
        """Trackbars Canny avec aperçu edges (pas de polylines)."""
        win = "INIT CANNY"
        cv2.namedWindow(win)
        cv2.createTrackbar("Low", win, self.canny_params["low"], 255, lambda x: None)
        cv2.createTrackbar("High", win, self.canny_params["high"], 255, lambda x: None)

        gray = cv2.cvtColor(frame_no_robot, cv2.COLOR_BGR2GRAY)
        while True:
            low = cv2.getTrackbarPos("Low", win)
            high = cv2.getTrackbarPos("High", win)
            edges = cv2.Canny(gray, low, high)
            cv2.imshow(win, edges)
            k = cv2.waitKey(1) & 0xFF
            if k in [13, ord(' ')]:
                self.canny_params["low"], self.canny_params["high"] = low, high
                break
            if k in [27, ord('q')]:
                break
        cv2.destroyWindow(win)

    # -----------------------------------------------------
    def init_polygons(self, frame_no_robot):
        """Trackbars min/max area avec aperçu polygones sur l'image (robot déjà blur)."""
        win = "INIT POLYGONS"
        cv2.namedWindow(win)
        cv2.createTrackbar("MinA", win, self.poly_params["min_area"], 50000, lambda x: None)
        cv2.createTrackbar("MaxA", win, self.poly_params["max_area"], 80000, lambda x: None)

        while True:
            self.poly_params["min_area"] = cv2.getTrackbarPos("MinA", win)
            self.poly_params["max_area"] = cv2.getTrackbarPos("MaxA", win)

            polys, _ = self.process(frame_no_robot, show_debug=False, skip_blur=True, log_pose=False)

            dbg = frame_no_robot.copy()
            for poly in polys:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(dbg, [pts], True, (0, 255, 0), 2)
            cv2.imshow(win, dbg)

            k = cv2.waitKey(1) & 0xFF
            if k in [13, ord(' ')]:
                break
            if k in [27, ord('q')]:
                break
        cv2.destroyWindow(win)

    # -----------------------------------------------------
    def process(self, frame, show_debug=False, skip_blur=False, log_pose=True):
        """
        Pipeline final sur une frame:
        - detect_robot ( red and green pr les couleurs)
        - blur_robot pour flouter robot
        -canny and fermeture morpho pr les trous
        - air min and max of polygone
        """
        st = self.detect_robot(frame)
        frame_clean = frame if skip_blur else self.blur_robot(frame, st)

        # si polygones figés (à l'init), je les renvoie directement
        if self.static_polys is not None:
            polys = list(self.static_polys)
        else:
            gray = cv2.cvtColor(frame_clean, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, self.canny_params["low"], self.canny_params["high"])
            edges = strengthen_edges(edges, ksize=4, iterations=4)  # augmente si traits restent séparés

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = []
            for c in contours:
                a = cv2.contourArea(c)
                if not (self.poly_params["min_area"] < a < self.poly_params["max_area"]):
                    continue

                epsilon = 0.01 * cv2.arcLength(c, True)  
                poly = cv2.approxPolyDP(c, epsilon, True)
                #poly = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
                pts = [(int(p[0][0]), int(p[0][1])) for p in poly]
                polys.append(pts)

        if log_pose:
            self._record_pose(st)

        if not show_debug:
            return polys, st

        dbg = frame_clean.copy()
        for poly in polys:
            cv2.polylines(dbg, [np.array(poly, dtype=np.int32)], True, (0, 255, 0), 2)
        if st["found"]:
            cv2.circle(dbg, st["red_center"], 6, (0, 0, 255), -1)
            cv2.circle(dbg, st["green_center"], 6, (0, 255, 0), -1)
            cv2.circle(dbg, st["center"], 6, (255, 255, 255), -1)
            th = st["theta"]
            x2 = int(st["center"][0] + 60 * np.cos(th))
            y2 = int(st["center"][1] + 60 * np.sin(th))
            cv2.arrowedLine(dbg, st["center"], (x2, y2), (255, 255, 255), 2)
        return polys, st, dbg

    # -----------------------------------------------------
    def initialize(self, frame):
        """
        Initialise tous les paramètres sur une seule image :
        1 robot color
        2) blur 
        3) Canny
        4) polygones detection
        """
        self.init_colors(frame)
        # après couleurs, je détecte + blur pour préparer les trackbars suivantes
        st = self.detect_robot(frame)
        frame_clean = self.blur_robot(frame, st)
        self.init_canny(frame_clean)
        self.init_polygons(frame_clean)

        # je fige les polygones dès l'init pour ne plus les recalculer ensuite
        polys, _ = self.process(frame, show_debug=False, skip_blur=False, log_pose=False)
        self.static_polys = list(polys)
        print("[INFO] Initialisation terminé ( couleurs + Canny + polygones figé).")

    # -----------------------------------------------------
    def get_pose_array(self, as_numpy=True, clear=False):
        """
        Retourne l'historique des poses sous forme de liste ou np.ndarray (n x 3).
        Utiliser clear=True pour vider l'historique après lecture.
        """
        if as_numpy:
            data = np.array(self.pose_history, dtype=np.float32)
            if data.size == 0:
                data = data.reshape(0, 3)
        else:
            data = self.pose_history
        if clear:
            self.pose_history = []
        return data

    def reset_pose_history(self):
        """Vide manuellement l'historique des poses."""
        self.pose_history = []


# =========================================================
# MAIN DEMO
# =========================================================
if __name__ == "__main__":
    """
    Utilisation :
    - mode_image : charge une image locale (relative -> dossier images/)
    - mode_webcam : ouvre la webcam
    L'initialisation se fait une seule fois au début (trackbars), puis on enchaîne
    la détection robot + obstacles. Pas de touches r/p, pas de recalibration en boucle.
    Paramètre show_debug pour afficher ou non le retour vision.
    """
    vision = Vision()
    show_debug = True           # mettre False si un autre module consomme juste les données
    mode_image = False           # False pour webcam
    image_name = "table10.jpg"   # change le nom si nécessaire

    if mode_image:
        frame0 = vision.load(image_name)
        vision.initialize(frame0)
        res = vision.process(frame0, show_debug=show_debug)
        if show_debug:
            _, _, dbg = res
            cv2.namedWindow("RESULT", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("RESULT", 1280, 720)
            h, w = dbg.shape[:2]
            if hasattr(cv2, "getWindowImageRect"):
                _, _, win_w, win_h = cv2.getWindowImageRect("RESULT")
            else:
                win_w, win_h = w, h
            scale = min(win_w / w, win_h / h)
            disp = cv2.resize(dbg, (max(1, int(w * scale)), max(1, int(h * scale))))
            cv2.imshow("RESULT", disp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            polys, st = res
            print("Robot:", st)
            print("Obstacles:", len(polys))
    else:
        # réglages capture (adapter si besoin)
        DESIRED_W, DESIRED_H = 1920, 1080
        DESIRED_FPS = 60
        FOURCC = "MJPG"
        AUTOFOCUS = 0       # 0=off, 1=on (si supporté)
        ISO = None          # mettez une valeur si supporté
        EXPOSURE = None     # mettez une valeur si supporté (souvent négatif)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la webcam.")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_H)
        cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)
        #cap.set(cv2.CAP_PROP_AUTOFOCUS, AUTOFOCUS)
        #cap.set(cv2.CAP_PROP_ISO_SPEED, ISO)
        #cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

        print(f"[INFO] FourCC demandé: {FOURCC}")
        rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        got_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Capture resolution ciblée: {rw}x{rh}")
        if got_fps > 0:
            print(f"[INFO] Capture FPS cible: {got_fps:.1f}")
        if hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
            print(f"[INFO] Autofocus={bool(cap.get(cv2.CAP_PROP_AUTOFOCUS))}")
        ok, frame0 = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Impossible de lire la webcam.")
        vision.initialize(frame0)

        if show_debug:
            cv2.namedWindow("RESULT", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("RESULT", 1280, 720)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = vision.process(frame, show_debug=show_debug)
            if show_debug:
                _, _, dbg = res
                h, w = dbg.shape[:2]
                if hasattr(cv2, "getWindowImageRect"):
                    _, _, win_w, win_h = cv2.getWindowImageRect("RESULT")
                else:
                    win_w, win_h = w, h
                scale = min(win_w / w, win_h / h)
                disp = cv2.resize(dbg, (max(1, int(w * scale)), max(1, int(h * scale))))
                cv2.imshow("RESULT", disp)
                if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                    break
            else:
                # si pas de debug, tu peux pousser les données ailleurs
                polys, st = res
                # exemple d'impression minimaliste
                print(f"Robot found={st['found']} | centre={st['center']} | obstacles={len(polys)}", end="\r")

        cap.release()
        cv2.destroyAllWindows()
