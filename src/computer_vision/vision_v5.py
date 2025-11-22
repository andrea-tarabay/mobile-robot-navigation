import cv2
import json
import numpy as np
import os

# ============================================
# UTILS
# ============================================

def hsv_mask_circular(hsv, hmin, hmax, smin=40, vmin=40):
    """
    Masque HSV en tenant compte du wrap 179 -> 0.
    hmin, hmax dans [0,179].
    Args: hsv (image HSV uint8), hmin/hmax/smin/vmin seuils.
    Return: masque binaire (uint8) où les pixels dans l'intervalle sont à 255.
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
    """
    4πA / P² pour mesurer si un contour est rond.
    Args: c contour opencv.
    Return: score de circularité (0..1+).
    """
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
    Args: mask binaire, aire min/max, circularité min.
    Return: liste de dicts pour chaque bulle compatible.
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
    Args: deux candidats ronds + paramètres de tolérance.
    Return: bool, True si la paire semble être le robot.
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
    Args: deux candidats ronds.
    Return: float (score relatif pour choisir la meilleure paire).
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
        """Init: états, caches, filtres EMA et monitoring pour la pipeline vision."""
        self.color_params = None   # HSV thresholds + aires
        self.canny_params = None   # Canny thresholds + aires

        # carte statique (obstacles) que je veux garder dès que le setup est figé
        self.use_static_map = False
        self.static_map_polys = None

        self.robot_state = {
            "found": False,
            "center": None,
            "red_center": None,
            "green_center": None,
            "theta": None,
            "red_area": 0.0,
            "green_area": 0.0,
            "robot_mask": None,  # masque final du robot (uint8)
            # filtres EMA pour lisser avant le Kalman / local path
            "smooth_center": None,
            "smooth_theta": None
        }
        #self.debug_remove_robot = 0
        self.debug_remove_robot = 1 # je laisse 1 quand je veux voir l'inpaint en live

        # filtre exponentiel pour lisser centre + orientation (je garde léger)
        self.pose_filter_alpha = 0.35
        self.pose_filtered = {"center": None, "theta": None}

        # stats basiques pour monitorer les frames perdues
        self.not_found_stats = {
            "frames": 0,
            "miss": 0,
            "warn_threshold": 0.25,   # 25% de frames manquées -> alerte
            "warn_min_frames": 30,    # j'attends un peu avant de spam un warning
            "last_warned_ratio": 0.0,
            "last_ratio": 0.0
        }

        BASE = os.path.dirname(os.path.abspath(__file__))
        self.IMGS = os.path.join(BASE, "images")
        # je garde un emplacement par défaut pour stocker les params calibrés
        self.default_params_path = os.path.join(BASE, "vision_params.json")

    def load(self, filename):
        """Charge une image depuis le dossier images interne. Args: filename (str). Return: image BGR."""
        path = os.path.join(self.IMGS, filename)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Impossible de charger {path}")
        return img

    # =======================================
    #  PERSISTENCE DES PARAMETRES (HSV + CANNY)
    # =======================================
    def save_params(self, filepath=None):
        """
        J'enregistre les params calibrés pour éviter de refaire les trackbars.
        - filepath None -> self.default_params_path
        Args: filepath str ou None.
        Return: None (écrit un JSON color_params/canny_params).
        """
        if self.color_params is None or self.canny_params is None:
            raise RuntimeError("Params incomplets : init_colors/init_canny d'abord.")

        if filepath is None:
            filepath = self.default_params_path

        blob = {
            "color_params": self.color_params,
            "canny_params": self.canny_params
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)
        print(f"[INFO] Params vision sauvegardés dans {filepath}")

    def load_params(self, filepath=None):
        """
        Recharge les seuils HSV + Canny depuis un fichier JSON.
        Je supporte les deux clés 'color_params' et 'canny_params'.
        Args: filepath str ou None.
        Return: dict chargé (et met à jour self.color_params/canny_params).
        """
        if filepath is None:
            filepath = self.default_params_path

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # tolérant sur les clés pour ne pas bloquer si le JSON est simple
        self.color_params = data.get("color_params") or data.get("color")
        self.canny_params = data.get("canny_params") or data.get("canny")

        if self.color_params is None or self.canny_params is None:
            raise ValueError(f"Fichier {filepath} ne contient pas color_params/canny_params")

        print(f"[INFO] Params vision chargés depuis {filepath}")
        return data

    # =======================================
    #  DETECTION ROBOT (STRICTE, OPTIMALE)
    # =======================================
    def detect_robot(self, frame, log_stats=True):
        """
        Détecte toutes les bulles rouges/vertes, cherche la meilleure paire
        respectant TES conditions.
        Si aucune paire valide -> found=False (préférence faux négatif).
        log_stats=False si je veux juste un masque (ex: remove_robot) sans suivre les stats.
        Args: frame BGR, log_stats (bool).
        Return: self.robot_state dict avec centres bruts/lissés, theta, aires, masque.
        """
        if self.color_params is None:
            self.robot_state.update({
                "found": False, "center": None,
                "red_center": None, "green_center": None,
                "theta": None, "red_area": 0.0, "green_area": 0.0,
                "robot_mask": None,
                "smooth_center": None,
                "smooth_theta": None
            })
            # je reset aussi le filtre EMA et j'update les stats de non-détection
            self._apply_pose_ema(None, None)
            if log_stats:
                self._update_not_found_stats(False)
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
                "robot_mask": None,
                "smooth_center": None,
                "smooth_theta": None
            })
            # je note la frame manquée pour le monitoring
            self._apply_pose_ema(None, None)
            if log_stats:
                self._update_not_found_stats(False)
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
        #print("Theta (deg):", np.degrees(theta))

        # masque robot = union des 2 contours, dilaté
        robot_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(robot_mask, [red["contour"]], -1, 255, -1)
        cv2.drawContours(robot_mask, [green["contour"]], -1, 255, -1)

        # dilatation pour couvrir tout le robot
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        robot_mask = cv2.dilate(robot_mask, k, iterations=1)

        # lissage EMA pour stabiliser avant d'envoyer au Kalman
        smooth_center, smooth_theta = self._apply_pose_ema(center, theta)

        self.robot_state.update({
            "found": True,
            "center": center,
            "red_center": red_center,
            "green_center": green_center,
            "theta": theta,
            "red_area": red["area"],
            "green_area": green["area"],
            "robot_mask": robot_mask,
            "smooth_center": smooth_center,
            "smooth_theta": smooth_theta
        })
        # j'update les stats uniquement quand le robot est trouvé
        if log_stats:
            self._update_not_found_stats(True)
        return self.robot_state

    def _apply_pose_ema(self, center, theta):
        """
        Petit filtre exponentiel maison pour lisser la pose et éviter les jumps.
        - centre lissé en (x,y) float
        - theta lissé via sin/cos pour respecter le wrap 2π
        Args: center (tuple x,y) ou None, theta (float rad) ou None.
        Return: (center_lissé int,int ou None, theta_lissé float ou None).
        """
        if center is None or theta is None:
            # si je perds le robot je repars de zéro (sinon on traîne un vieux état)
            self.pose_filtered = {"center": None, "theta": None}
            return None, None

        alpha = float(self.pose_filter_alpha)
        cx, cy = float(center[0]), float(center[1])

        if self.pose_filtered["center"] is None:
            # première détection -> pas de lissage
            filt_center = (cx, cy)
            filt_theta = float(theta)
        else:
            px, py = self.pose_filtered["center"]
            filt_center = (
                (1.0 - alpha) * px + alpha * cx,
                (1.0 - alpha) * py + alpha * cy
            )
            # je lisse l'angle en combinant sin/cos (évite les sauts autour de pi/-pi)
            prev_theta = self.pose_filtered["theta"]
            filt_theta = np.arctan2(
                (1.0 - alpha) * np.sin(prev_theta) + alpha * np.sin(theta),
                (1.0 - alpha) * np.cos(prev_theta) + alpha * np.cos(theta)
            )

        self.pose_filtered["center"] = filt_center
        self.pose_filtered["theta"] = filt_theta

        # je renvoie un centre arrondi (int) pour le dessin + theta float pour le planner
        return (int(filt_center[0]), int(filt_center[1])), float(filt_theta)

    def _update_not_found_stats(self, found):
        """
        Je cumule les frames ratées pour savoir si la détection est trop bruyante.
        Si le ratio > warn_threshold, j'affiche un warning console pour recalibrer.
        Args: found (bool) indique si la frame courante a détecté.
        Return: ratio courant de frames ratées.
        """
        stats = self.not_found_stats
        stats["frames"] += 1
        if not found:
            stats["miss"] += 1

        ratio = stats["miss"] / max(1, stats["frames"])
        stats["last_ratio"] = ratio

        # évite de spammer : je n'alerte que si le ratio dépasse le seuil de manière nouvelle
        if (stats["frames"] >= stats["warn_min_frames"]
                and ratio > stats["warn_threshold"]
                and ratio - stats["last_warned_ratio"] > 0.02):
            print(f"[WARN] Robot non trouvé sur {ratio*100:.1f}% des frames "
                  f"(>{stats['warn_threshold']*100:.0f}%). Recalibre HSV/Canny ou ajuste l'éclairage.")
            stats["last_warned_ratio"] = ratio

        return ratio

    def reset_monitoring(self):
        """Je repars de zéro pour les stats de non-détection (utile avant un nouveau run). Return: None."""
        self.not_found_stats.update({
            "frames": 0,
            "miss": 0,
            "last_ratio": 0.0,
            "last_warned_ratio": 0.0
        })

    # =======================================
    #  INITIALISATION COULEUR (ROBOT)
    # =======================================
    def init_colors(self, frame):
        """Trackbars pour calibrer HSV. Args: frame BGR (image fixe). Return: None (stocke self.color_params)."""
        cv2.namedWindow("COLOR INIT")

        R_hmin, R_hmax = 170, 10
        G_hmin, G_hmax = 40, 90
        minA, maxA = 50, 2000   # je reste large pour attraper le robot même s'il s'éloigne

        # trackbars = mon outil manuel pour caler les seuils HSV
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

            # ici je ne veux pas polluer les stats de monitoring
            state = self.detect_robot(frame, log_stats=False)

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
        Args: frame BGR.
        Return: (image nettoyée, masque robot) ; si pas trouvé -> (frame copy, None).
        
        Si self.debug_remove_robot == 1 :
            affiche l'image nettoyée (pour la documentation)
        """
        state = self.detect_robot(frame, log_stats=False)
        if not state["found"] or state["robot_mask"] is None:
            cleaned = frame.copy()

            # --- DEBUG OPTIONNEL ---
            if getattr(self, "debug_remove_robot", 0) == 1:
                cv2.imshow("REMOVE_ROBOT_DEBUG", cleaned)
                cv2.waitKey(1)

            return cleaned, None

        mask = state["robot_mask"]

        # inpaint pour virer complètement le robot avant Canny
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
        """Prépare l'image pour Canny (robot supprimé, CLAHE, bilateral). Args: frame BGR. Return: (clean, smooth, robot_mask)."""
        clean, robot_mask = self.remove_robot(frame)

        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)  # j'augmente le contraste local pour mieux choper les edges

        # même filtre partout, le bilateral garde les edges tout en virant le bruit
        smooth = cv2.bilateralFilter(eq, d=9, sigmaColor=50, sigmaSpace=50)
        return clean, smooth, robot_mask

    # =======================================
    #  INITIALISATION CANNY (identique final)
    # =======================================
    def init_canny(self, frame):
        """Trackbars pour calibrer Canny/areas. Args: frame BGR (image fixe). Return: None (stocke self.canny_params)."""
        cv2.namedWindow("CANNY INIT")

        low, high = 70, 120
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

            # affichage basé sur l'image CANNNY (je vois vite si je coupe trop/peu)
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
    #  EXTRACTION POLYGONES COMMUNE
    # =======================================
    def _extract_polygons(self, frame):
        """
        Pipeline complet Canny + contours, sans caching.
        Je factorise pour réutiliser en init, freeze et calcul live.
        Args: frame BGR.
        Return: liste de polygones (list[list[(x,y)]]).
        """
        _clean, smooth, robot_mask = self._preprocess_for_canny(frame)

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

    def freeze_map_from_frame(self, frame):
        """
        J'extrais la map une fois (image fixe de la webcam au setup).
        Ensuite self.use_static_map permet de ne plus recalculer les polygones
        à chaque frame live.
        Args: frame BGR (image de référence).
        Return: polygones détectés (list de listes de points).
        """
        if self.canny_params is None:
            raise RuntimeError("Canny non initialisé : appelle init_canny() d'abord.")

        polys = self._extract_polygons(frame)
        self.static_map_polys = list(polys)
        self.use_static_map = True
        return polys

    def clear_static_map(self):
        """Je repasse en mode recalcul temps réel (utile si la map change). Return: None."""
        self.static_map_polys = None
        self.use_static_map = False

    # =======================================
    #  DÉTECTION POLYGONES FINAL (cache map statique)
    # =======================================
    def process(self, frame):
        """
        Détection de polygones obstacles avec cache statique.
        Args: frame BGR.
        Return: liste de polygones (list de points).
        """
        if self.canny_params is None:
            raise RuntimeError("Canny non initialisé : appelle init_canny() d'abord.")

        # si j'ai figé la map à l'init, je la renvoie telle quelle
        if self.use_static_map and self.static_map_polys is not None:
            return list(self.static_map_polys)

        # sinon calcul normal sur l'image courante
        polys = self._extract_polygons(frame)

        # si le mode statique est activé mais pas encore peuplé, je le remplis
        if self.use_static_map:
            self.static_map_polys = list(polys)

        return polys

    # =======================================
    #  PIPELINE WEBCAM LIVE
    # =======================================
    def run_webcam(self, cam_index=0, params_path=None, warn_ratio=None):
        """
        Pipeline complet webcam que je veux utiliser pendant les tests:
        1) grab 1 frame -> init (ou chargement) des params HSV/Canny
        2) freeze_map_from_frame(frame_init) pour séparer obstacles/robot
        3) boucle: detect_robot(frame_live) + affichage, sans recalcul map
        Touches utiles:
            - 'r' : refreeze la map si la scène a bougé (clear_static_map + freeze)
            - 'p' : relance les trackbars pour recalibrer + sauvegarde des params
            - 'q' ou ESC : quitte proprement
        warn_ratio: ratio de frames ratées qui déclenche le warning console (None => seuil actuel)
        Args: cam_index (int), params_path (str JSON), warn_ratio (float).
        Return: None (boucle jusqu'à sortie).
        """
        if params_path is None:
            params_path = self.default_params_path

        if warn_ratio is not None:
            # je peux ajuster le seuil d'alerte sans toucher au reste du code
            self.not_found_stats["warn_threshold"] = float(warn_ratio)

        self.reset_monitoring()

        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la webcam index={cam_index}")

        ret, frame_init = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Impossible de lire la première frame webcam.")

        # ==== CHARGEMENT / INITIALISATION DES PARAMS ====
        try:
            if params_path and os.path.exists(params_path):
                print(f"[INFO] Chargement des params depuis {params_path}")
                self.load_params(params_path)
            else:
                print("[INFO] Pas de params préexistants -> trackbars init")
                self.init_colors(frame_init)
                self.init_canny(frame_init)
                if params_path:
                    self.save_params(params_path)
        except Exception as e:
            # si le JSON est corrompu je relance une init propre
            print(f"[WARN] Chargement params échoué ({e}), je relance une init trackbars.")
            self.init_colors(frame_init)
            self.init_canny(frame_init)
            if params_path:
                self.save_params(params_path)

        # ==== MAP FIGEE ====
        self.freeze_map_from_frame(frame_init)
        print("[INFO] Map figée (polygones obstacles) à partir de la frame d'init.")

        # ==== BOUCLE LIVE ====
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame webcam manquante, arrêt boucle.")
                break

            # process() renvoie directement la map figée si elle existe
            polys = self.process(frame)
            robot = self.detect_robot(frame)

            view = frame.copy()

            # affichage des obstacles (verts) pour debug path planning
            for poly in polys:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(view, [pts], True, (0, 255, 0), 2)

            # overlay du robot + version lissée pour montrer la stabilité
            if robot["found"]:
                cx, cy = robot["center"]
                cv2.circle(view, (cx, cy), 6, (255, 255, 255), -1)
                th = robot["theta"]
                x2 = int(cx + 60*np.cos(th))
                y2 = int(cy + 60*np.sin(th))
                cv2.arrowedLine(view, (cx, cy), (x2, y2), (0, 0, 255), 3, tipLength=0.3)

                # en jaune : la version lissée (celle que je veux pousser au planner)
                if robot["smooth_center"] is not None:
                    scx, scy = robot["smooth_center"]
                    cv2.circle(view, (scx, scy), 6, (0, 255, 255), 2)
                    sth = robot["smooth_theta"]
                    sx2 = int(scx + 70*np.cos(sth))
                    sy2 = int(scy + 70*np.sin(sth))
                    cv2.arrowedLine(view, (scx, scy), (sx2, sy2), (0, 255, 255), 2, tipLength=0.25)

                cv2.putText(view, "Robot: FOUND", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(view, "Robot: NOT FOUND", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # monitoring du taux de frames ratées
            miss_ratio = self.not_found_stats["last_ratio"] * 100.0
            cv2.putText(view, f"miss={miss_ratio:.0f}%", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            cv2.imshow("VISION LIVE", view)
            key = cv2.waitKey(1) & 0xFF

            if key in [27, ord('q')]:
                print("[INFO] Sortie demandée (q/ESC).")
                break

            if key == ord('r'):
                # je mets à jour la map statique avec la frame courante
                print("[INFO] Re-freeze de la map (touche r).")
                self.clear_static_map()
                self.freeze_map_from_frame(frame)

            if key == ord('p'):
                # recalibration complète en live
                print("[INFO] Recalibration params (touche p).")
                self.init_colors(frame)
                self.init_canny(frame)
                if params_path:
                    self.save_params(params_path)
                self.freeze_map_from_frame(frame)
                self.reset_monitoring()

        cap.release()
        cv2.destroyAllWindows()

    # =======================================
    #  ACCÈS SIMPLE AU POSE ROBOT
    # =======================================
    def get_robot_pose(self, frame):
        """
        Retourne (found, center(x,y), theta_rad).
        Utilise detect_robot().
        Args: frame BGR.
        Return: tuple (found, center, theta).
        """
        st = self.detect_robot(frame)
        return st["found"], st["center"], st["theta"]


# ============================================
# EXEMPLE IMAGE FIXE
# ============================================
if __name__ == "__main__":

    vision = Vision()
    img = vision.load("table7.jpg")
    # si je veux tester en live : vision.run_webcam(cam_index=0, params_path=vision.default_params_path)  # warn_ratio=0.3 pour alerte plus tôt

    print("=== INIT COULEURS ===")
    vision.init_colors(img)

    print("=== INIT CANNY ===")
    vision.init_canny(img)

    # exemple offline : je fige la map une fois sur l'image, comme en webcam
    polys = vision.freeze_map_from_frame(img)

    robot = vision.detect_robot(img)  # robot["center"], robot["theta"], robot["found"]

    print("Robot flag:", int(robot["found"]))
    print("Robot center:", robot["center"])
    print("Robot theta (rad):", robot["theta"])
    print("Robot theta (deg):", np.degrees(robot["theta"]))

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
