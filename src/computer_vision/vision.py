import cv2
import numpy as np




# ---------------------------------------------------------
# INITIALISATION DES PARAMÈTRES AVEC SLIDERS
# ---------------------------------------------------------
def initialisation(frame):
    import cv2
    import numpy as np

    # Valeurs de base
    init_low = 30
    init_high = 120
    init_min_area = 300
    init_max_robot = 5000

    # Fenêtre sliders
    cv2.namedWindow("INITIALISATION")

    cv2.createTrackbar("Canny Low",  "INITIALISATION", init_low, 255, lambda x: None)
    cv2.createTrackbar("Canny High", "INITIALISATION", init_high, 255, lambda x: None)
    cv2.createTrackbar("Min Area",   "INITIALISATION", init_min_area, 10000, lambda x: None)
    cv2.createTrackbar("Max Robot",  "INITIALISATION", init_max_robot, 20000, lambda x: None)

    while True:
        # Lire sliders
        low  = cv2.getTrackbarPos("Canny Low",  "INITIALISATION")
        high = cv2.getTrackbarPos("Canny High", "INITIALISATION")
        min_area = cv2.getTrackbarPos("Min Area", "INITIALISATION")
        max_robot = cv2.getTrackbarPos("Max Robot", "INITIALISATION")

        # Images intermédiaires
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, low, high)

        # Détection contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # image pour debug
        debug = frame.copy()
        robot = None
        max_area_found = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            eps = 0.02 * cv2.arcLength(c, True)
            poly = cv2.approxPolyDP(c, eps, True)
            pts = np.array(poly, dtype=np.int32)

            # robot ?
            if min_area < area < max_robot and area > max_area_found:
                max_area_found = area
                M = cv2.moments(c)
                if M["m00"] != 0:
                    robot = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                color = (255,0,0)
            else:
                color = (0,255,0)

            cv2.polylines(debug, [pts], True, color, 2)

        # Dessiner robot si trouvé
        if robot is not None:
            cv2.circle(debug, robot, 10, (255,0,0), -1)
            cv2.putText(debug, "ROBOT", (robot[0]+10, robot[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # afficher
        cv2.imshow("INITIALISATION", debug)
        cv2.imshow("CANNY", edges)

        # ENTER = valider / q = quitter
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord(' '):  # ENTER ou espace
            break
        if key == 27 or key == ord('q'):  # ESC
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return {
        "canny_low": low,
        "canny_high": high,
        "min_area": min_area,
        "max_robot_area": max_robot
    }






class Vision:
    def __init__(self):
        pass

    def load(self, path="table.jpg"):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Impossible de charger {path}")
        return img

    def get_state(self, frame):

        # ===== 1. GRAY =====
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("1 - Gray", gray)

        # ===== 2. BLUR =====
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        cv2.imshow("2 - Blur", blur)

        # ===== 3. CANNY =====
        edges = cv2.Canny(blur, self.params["canny_low"], self.params["canny_high"])
        cv2.imshow("3 - Canny", edges)

        # ===== 4. CONTOURS =====
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # image pour debug contours
        contour_debug = frame.copy()
        cv2.drawContours(contour_debug, contours, -1, (255,0,255), 2)
        cv2.imshow("4 - Contours bruts", contour_debug)

        obstacles = []
        robot = None
        max_robot_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.params["min_area"]:     # <-- Ajuste pour ignorer le bruit
                continue

            eps = 0.02 * cv2.arcLength(c, True)
            poly = cv2.approxPolyDP(c, eps, True)
            pts = [(int(p[0][0]), int(p[0][1])) for p in poly]

            # Robot = petit contour
            if self.params["min_area"] < area < self.params["max_robot_area"]:
                max_robot_area = area
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    robot = (cx, cy)
                continue

            obstacles.append(pts)

        return {
            "robot": robot,
            "obstacles": obstacles,
            "rewards": []
        }

    def draw(self, frame, state):
        out = frame.copy()

        # Robot
        if state["robot"] is not None:
            x, y = state["robot"]
            cv2.circle(out, (int(x), int(y)), 10, (255, 0, 0), -1)
            cv2.putText(out, "ROBOT", (int(x)+10, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # Obstacles
        for poly in state["obstacles"]:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(out, [pts], True, (0,255,0), 3)
            for (px, py) in poly:
                cv2.circle(out, (px, py), 5, (0,255,0), -1)

        return out


# =========================================================
# UTILISATION
# =========================================================

vision = Vision()

img = vision.load("table3.jpg")

# ----- INITIALISATION -----
vision.params = initialisation(img)

# ----- FONCTION NORMALE -----
state = vision.get_state(img)
res = vision.draw(img, state)

cv2.imshow("5 - RESULTAT FINAL", res)

# ======== FIX MAC (boucle non bloquante) ============
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC ou q pour quitter
        break

cv2.destroyAllWindows()
cv2.waitKey(1)