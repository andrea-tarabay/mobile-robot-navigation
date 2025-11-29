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


def robot_pair_ok(red, green, max_r_rel_diff=0.25, dist_factor=4.0):
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
        self.pose_filter_alpha = 0.45
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

        # perspective / redressement : je stocke la matrice pour warpPerspective
        self.warp_matrix = None
        self.warp_size = None  # (width, height)

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
        Return: None (écrit un JSON color_params/canny_params + warp éventuel).
        """
        if self.color_params is None or self.canny_params is None:
            raise RuntimeError("Params incomplets : init_colors/init_canny d'abord.")

        if filepath is None:
            filepath = self.default_params_path

        blob = {
            "color_params": self.color_params,
            "canny_params": self.canny_params,
            "warp_matrix": self.warp_matrix.tolist() if self.warp_matrix is not None else None,
            "warp_size": self.warp_size
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
        warp_m = data.get("warp_matrix")
        warp_s = data.get("warp_size")
        if warp_m is not None:
            self.warp_matrix = np.array(warp_m, dtype=np.float32)
            self.warp_size = tuple(warp_s) if warp_s is not None else None

        if self.color_params is None or self.canny_params is None:
            raise ValueError(f"Fichier {filepath} ne contient pas color_params/canny_params")

        print(f"[INFO] Params vision chargés depuis {filepath}")
        return data

    # =======================================
    #  PERSPECTIVE / REDRESSEMENT (markers 4 coins)
    # =======================================
    def _apply_perspective(self, frame):
        """
        Applique la matrice de warp si elle existe. Sinon renvoie la frame telle quelle.
        Args: frame BGR.
        Return: frame BGR redressée.
        """
        if self.warp_matrix is None or self.warp_size is None:
            return frame
        return cv2.warpPerspective(frame, self.warp_matrix, self.warp_size)

    def _order_points_tlbr(self, pts):
        """
        Ordonne 4 points en (top-left, top-right, bottom-right, bottom-left)
        en utilisant x+y (min/max) et x-y (min/max).
        Args: pts iterable de 4 (x,y).
        Return: np.float32(4,2) ordonné.
        """
        pts = np.array(pts, dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]   # top-left
        ordered[2] = pts[np.argmax(s)]   # bottom-right
        ordered[1] = pts[np.argmin(diff)]  # top-right
        ordered[3] = pts[np.argmax(diff)]  # bottom-left
        return ordered

    def calibrate_perspective(self, frame):
        """
        Cherche 4 marqueurs rectangulaires (polygones) aux coins de la table,
        calcule la matrice de redressement pour que ces marqueurs collent aux 4 coins de l'image.
        Hypothèse: les 4 plus gros polygones détectés (via Canny) sont ces marqueurs.
        Args: frame BGR inclinée.
        Return: (warp_matrix, warp_size, src_pts, dst_pts) pour debug.
        """
        h, w = frame.shape[:2]

        raise RuntimeError("Calibration perspective automatique retirée (utilise calibrate_perspective_manual à la place).")

    def calibrate_perspective_manual(self, frame):
        """
        Version manuelle : je place/déplace 4 points (coins de la table) à la souris.
        Les points initiaux sont aux coins de l'image; je calcule une homographie vers un rectangle.
        Return: (warp_matrix, warp_size, src_pts, dst_pts) pour debug.
        """
        h, w = frame.shape[:2]
        window = "PERSPECTIVE MANUAL"

        # points init (marges légères)
        pts = np.array([
            [20, 20],
            [w - 20, 20],
            [w - 20, h - 20],
            [20, h - 20]
        ], dtype=np.float32)
        dragging = {"idx": None}

        def on_mouse(event, x, y, flags, param):
            # petit handler pour déplacer le point le plus proche
            if event == cv2.EVENT_LBUTTONDOWN:
                dists = np.linalg.norm(pts - np.array([x, y], dtype=np.float32), axis=1)
                idx = int(np.argmin(dists))
                if dists[idx] < 40:  # seuil de sélection
                    dragging["idx"] = idx
            elif event == cv2.EVENT_MOUSEMOVE and dragging["idx"] is not None:
                pts[dragging["idx"]] = [x, y]
            elif event == cv2.EVENT_LBUTTONUP:
                dragging["idx"] = None

        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_mouse)

        print("[INFO] Place les 4 points aux coins de la table (glisser). ENTER/SPACE pour valider, r pour reset, q/ESC pour annuler.")

        while True:
            vis = frame.copy()
            # dessiner les points et le quadrilatère
            for i, p in enumerate(pts):
                cv2.circle(vis, (int(p[0]), int(p[1])), 8, (0, 255, 255), -1)
                cv2.putText(vis, f"{i}", (int(p[0])+5, int(p[1])-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.polylines(vis, [pts.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(vis, "Drag points. ENTER=OK, r=reset, q/ESC=cancel",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window, vis)

            k = cv2.waitKey(20) & 0xFF
            if k in [13, ord(' ')]:  # ENTER / space
                break
            if k in [ord('r')]:
                pts = np.array([
                    [20, 20],
                    [w - 20, 20],
                    [w - 20, h - 20],
                    [20, h - 20]
                ], dtype=np.float32)
            if k in [27, ord('q')]:  # ESC / q -> cancel
                cv2.destroyWindow(window)
                raise RuntimeError("Calibration manuelle annulée.")

            # touche 'a' pour auto-reset si besoin
            if k in [ord('a')]:
                dragging["idx"] = None

        cv2.destroyWindow(window)

        # destination rectangle basé sur les longueurs max haut/bas et gauche/droite
        width_top = np.linalg.norm(pts[1] - pts[0])
        width_bottom = np.linalg.norm(pts[2] - pts[3])
        height_left = np.linalg.norm(pts[3] - pts[0])
        height_right = np.linalg.norm(pts[2] - pts[1])
        width_dst = int(max(width_top, width_bottom))
        height_dst = int(max(height_left, height_right))
        width_dst = max(width_dst, 50)
        height_dst = max(height_dst, 50)

        dst = np.array([
            [0, 0],
            [width_dst - 1, 0],
            [width_dst - 1, height_dst - 1],
            [0, height_dst - 1]
        ], dtype=np.float32)

        self.warp_matrix = cv2.getPerspectiveTransform(pts, dst)
        self.warp_size = (width_dst, height_dst)

        print("[INFO] Perspective calibrée manuellement (warp activé).")
        print("     src:", pts.tolist())
        print("     dst:", dst.tolist())

        return self.warp_matrix, self.warp_size, pts, dst

    # =======================================
    #  DETECTION ROBOT (STRICTE, OPTIMALE)
    # =======================================
    def detect_robot(self, frame, log_stats=True, apply_warp=True):
        """
        Détecte toutes les bulles rouges/vertes, cherche la meilleure paire
        respectant TES conditions.
        Si aucune paire valide -> found=False (préférence faux négatif).
        log_stats=False si je veux juste un masque (ex: remove_robot) sans suivre les stats.
        Args: frame BGR, log_stats (bool), apply_warp (bool) applique ou non la perspective.
        Return: self.robot_state dict avec centres bruts/lissés, theta, aires, masque.
        """
        if apply_warp:
            frame = self._apply_perspective(frame)

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
                if not robot_pair_ok(r, g, max_r_rel_diff=0.25, dist_factor=4): #distance centres <= sqrt(min(area)) * dist_factor
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
        frame_warp = self._apply_perspective(frame)
        state = self.detect_robot(frame_warp, log_stats=False, apply_warp=False)
        if not state["found"] or state["robot_mask"] is None:
            cleaned = frame_warp.copy()

            # --- DEBUG OPTIONNEL ---
            if getattr(self, "debug_remove_robot", 0) == 1:
                cv2.imshow("REMOVE_ROBOT_DEBUG", cleaned)
                cv2.waitKey(1)

            return cleaned, None

        mask = state["robot_mask"]

        # inpaint pour virer complètement le robot avant Canny
        removed = cv2.inpaint(frame_warp, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

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
        cv2.createTrackbar("Max Area",   "CANNY INIT", maxA, 2000000, lambda x: None)

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

        # petite fermeture morpho pour boucher les trous 1 px (évite les polygones ouverts)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))#ici Canny matrix probleme
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)

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
    #  INITIALISATION COMPLETE (une seule fois)
    # =======================================
    def initialize_session(self, frame, params_path=None, force_trackbars=False,
                           manual_warp=True, save_params=True):
        """
        Pipeline d'init unique et simple:
        1) charge params existants OU relance les trackbars si force_trackbars ou pas de fichier
        2) si manual_warp: ouvre un GUI pour placer 4 points de perspective
        3) fige la map d'obstacles (polygones) une fois pour toutes
        4) sauvegarde les paramètres (HSV/Canny/warp) si demandé
        Args:
            frame: image BGR de référence (fixe)
            params_path: chemin JSON (None -> default_params_path)
            force_trackbars: True pour recalibrer HSV/Canny même si un fichier existe
            manual_warp: True pour lancer l'UI de placement des 4 coins (sinon pas de warp)
            save_params: True pour écrire le JSON après init
        Return: polygones détectés (map statique).
        """
        self.reset_monitoring()
        if params_path is None:
            params_path = self.default_params_path

        # 1) d'abord, je calibre le warp si demandé (sur l'image brute)
        if manual_warp:
            try:
                self.calibrate_perspective_manual(frame)
            except Exception as e:
                print(f"[WARN] Warp manuel non appliqué: {e}")

        # 2) je travaille ensuite sur l'image redressée pour toutes les trackbars/map
        frame_ref = self._apply_perspective(frame)

        need_trackbars = force_trackbars or (params_path and not os.path.exists(params_path))
        if need_trackbars:
            print("[INFO] Trackbars init (pas de params ou recalibration forcée).")
            self.init_colors(frame_ref)
            self.init_canny(frame_ref)
        else:
            print(f"[INFO] Chargement des params existants ({params_path}).")
            try:
                self.load_params(params_path)
            except Exception as e:
                print(f"[WARN] Chargement params échoué ({e}), je bascule en trackbars.")
                self.init_colors(frame_ref)
                self.init_canny(frame_ref)

        # map obstacles figée (calculée une seule fois)
        polys = self.freeze_map_from_frame(frame_ref)

        # sauvegarde des paramètres (y compris warp)
        if save_params and params_path:
            try:
                self.save_params(params_path)
            except Exception as e:
                print(f"[WARN] Sauvegarde params échouée: {e}")

        return polys

    # =======================================
    #  PIPELINE WEBCAM LIVE
    # =======================================
    def run_webcam(self, cam_index=0, params_path=None, warn_ratio=None, force_new_init=False):
        """
        Pipeline complet webcam que je veux utiliser pendant les tests:
        1) grab 1 frame -> init (ou chargement) des params HSV/Canny + warp manuel + map
        2) freeze_map_from_frame(frame_init) pour séparer obstacles/robot (calculé une seule fois)
        3) boucle: detect_robot(frame_live) + affichage, sans recalcul map
        Touches utiles:
            - 'r' : refreeze la map si la scène a bougé (clear_static_map + freeze)
            - 'p' : relance les trackbars pour recalibrer + sauvegarde des paramsr
            - 'q' ou ESC : quitte proprement
        warn_ratio: ratio de frames ratées qui déclenche le warning console (None => seuil actuel)
        Args: cam_index (int), params_path (str JSON), warn_ratio (float), force_new_init (bool).
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

        # ==== INITIALISATION UNIQUE (trackbars + warp manuel + map figée) ====
        self.initialize_session(
            frame_init,
            params_path=params_path,
            force_trackbars=force_new_init,
            manual_warp=True,
            save_params=True
        )

        # ==== BOUCLE LIVE ====
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame webcam manquante, arrêt boucle.")
                break

            # je prépare une version redressée pour affichage
            view = self._apply_perspective(frame.copy())
            cv2.imwrite("vision_live_output_pour_mehdi.jpg", view)

            # process() renvoie directement la map figée si elle existe
            polys = self.process(frame)
            robot = self.detect_robot(frame)

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
                self.initialize_session(
                    frame,
                    params_path=params_path,
                    force_trackbars=True,
                    manual_warp=True,
                    save_params=True
                )

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

    # =======================================
    #  DEMO IMAGE FIXE (pour screenshots / rendu)
    # =======================================
    def demo_image(self, img_path, force_trackbars=True, manual_warp=True):
        """
        Charge une image, fait l'init complète (trackbars si demandé), puis affiche la map + robot.
        Utile pour capturer des screenshots pour le rapport.
        - img_path relatif : cherché d'abord dans src/computer_vision/images via self.load()
        - img_path absolu  : chargé directement avec cv2.imread
        """
        # je passe par self.load pour supporter le dossier images interne
        if os.path.isabs(img_path):
            img = cv2.imread(img_path)
        else:
            try:
                img = self.load(img_path)
            except FileNotFoundError:
                # fallback: essayer relatif au cwd si l'utilisateur fournit un autre répertoire
                img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Impossible de charger {img_path}")

        polys = self.initialize_session(
            img,
            params_path=None,          # pas d'écriture de fichier par défaut
            force_trackbars=force_trackbars,
            manual_warp=manual_warp,
            save_params=False
        )

        robot = self.detect_robot(img)

        out = self._apply_perspective(img.copy())
        for poly in polys:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 0), 3)

        if robot["found"]:
            cx, cy = robot["center"]
            cv2.circle(out, (cx, cy), 7, (255, 255, 255), -1)
            th = robot["theta"]
            x2 = int(cx + 60*np.cos(th))
            y2 = int(cy + 60*np.sin(th))
            cv2.arrowedLine(out, (cx, cy), (x2, y2), (0, 0, 255), 3, tipLength=0.3)

        cv2.imshow("DEMO IMAGE", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return polys, robot


# ============================================
# EXEMPLE IMAGE FIXE
# ============================================
if __name__ == "__main__":

    vision = Vision()

    # 1) Mode démo sur image fixe (screenshots pour le rendu)
    #vision.demo_image("table7.jpg", force_trackbars=True, manual_warp=True)

    # 2) Mode live webcam (calibration manuelle 4 points + trackbars si besoin)
    vision.run_webcam(cam_index=0, params_path=vision.default_params_path, force_new_init=True)


    
