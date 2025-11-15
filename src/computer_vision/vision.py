import cv2
import numpy as np


# ---------------------------------------------------------
# INITIALISATION DES PARAMÈTRES AVEC SLIDERS
# ---------------------------------------------------------
def initialisation(frame):

    init_low = 30
    init_high = 120
    init_min_area = 300
    init_max_area = 20000

    cv2.namedWindow("INITIALISATION")
    cv2.namedWindow("CANNY")

    cv2.createTrackbar("Canny Low",  "INITIALISATION", init_low, 255, lambda x: None)
    cv2.createTrackbar("Canny High", "INITIALISATION", init_high, 255, lambda x: None)
    cv2.createTrackbar("Min Area",   "INITIALISATION", init_min_area, 20000, lambda x: None)
    cv2.createTrackbar("Max Area",   "INITIALISATION", init_max_area, 50000, lambda x: None)

    while True:

        low  = cv2.getTrackbarPos("Canny Low",  "INITIALISATION")
        high = cv2.getTrackbarPos("Canny High", "INITIALISATION")
        min_area = cv2.getTrackbarPos("Min Area", "INITIALISATION")
        max_area = cv2.getTrackbarPos("Max Area", "INITIALISATION")

        # pipeline simple
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(gray, (5,5), 0)
        #edges = cv2.Canny(blur, low, high)

        # 2) Renforcer les contrastes
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        eq = clahe.apply(gray)

        # 3) Lissage intelligent (préserve les bords)
        smooth = cv2.bilateralFilter(eq, d=7, sigmaColor=50, sigmaSpace=50)

        # 4) Canny avec sliders
        edges = cv2.Canny(smooth, low, high)


        # contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # afficher basÉ sur edges pour que Canny soit visible
        debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                eps = 0.02 * cv2.arcLength(c, True)
                poly = cv2.approxPolyDP(c, eps, True)
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(debug, [pts], True, (0,255,0), 2)

        cv2.imshow("INITIALISATION", debug)
        cv2.imshow("CANNY", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord(' '):  # ENTER
            break
        if key == 27 or key == ord('q'):  # ESC
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return {
        "canny_low": low,
        "canny_high": high,
        "min_area": min_area,
        "max_area": max_area
    }



# ---------------------------------------------------------
# VISION
# ---------------------------------------------------------
class Vision:
    def __init__(self):
        pass

    def load(self, path="table3.jpg"):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Impossible de charger {path}")
        return img

    def get_state(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(gray, (5,5), 0)
        #edges = cv2.Canny(blur, self.params["canny_low"], self.params["canny_high"])

        # 2) Renforcer les contrastes
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        eq = clahe.apply(gray)

        # 3) Lissage intelligent (préserve les bords)
        smooth = cv2.bilateralFilter(eq, d=7, sigmaColor=50, sigmaSpace=50)

        # 4) Canny avec sliders
        edges = cv2.Canny(
        smooth,
        self.params["canny_low"],
        self.params["canny_high"]
)

        #cv2.imshow("1 - Gray", gray)
        ##cv2.imshow("2 - Blur", blur)
        cv2.imshow("3 - Canny", edges)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_debug = frame.copy()
        cv2.drawContours(contour_debug, contours, -1, (255,0,255), 2)
        cv2.imshow("4 - Contours bruts", contour_debug)

        polys = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.params["min_area"] or area > self.params["max_area"]:
                continue

            eps = 0.02 * cv2.arcLength(c, True)
            poly = cv2.approxPolyDP(c, eps, True)
            pts = [(int(p[0][0]), int(p[0][1])) for p in poly]

            polys.append(pts)

        return polys

    def draw(self, frame, polys):
        out = frame.copy()

        for poly in polys:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(out, [pts], True, (0,255,0), 3)
            for (px, py) in poly:
                cv2.circle(out, (px, py), 5, (0,255,0), -1)

        return out



# ---------------------------------------------------------
# UTILISATION
# ---------------------------------------------------------
vision = Vision()
img = vision.load("table2.jpg")

vision.params = initialisation(img)

polys = vision.get_state(img)
res = vision.draw(img, polys)

cv2.imshow("RESULTAT FINAL", res)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
cv2.waitKey(1)