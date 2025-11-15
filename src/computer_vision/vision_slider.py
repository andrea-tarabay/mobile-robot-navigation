import cv2
import numpy as np

def nothing(x):
    pass

# charge l'image
img = cv2.imread("table2.jpg")
if img is None:
    raise FileNotFoundError("Image introuvable")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# fenetre sliders
cv2.namedWindow("Sliders")
cv2.createTrackbar("Low",  "Sliders", 30, 255, nothing)
cv2.createTrackbar("High", "Sliders", 120, 255, nothing)

while True:
    low  = cv2.getTrackbarPos("Low",  "Sliders")
    high = cv2.getTrackbarPos("High", "Sliders")

    # Canny
    edges = cv2.Canny(blur, low, high)

    # Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug = img.copy()
    cv2.drawContours(debug, contours, -1, (0,255,0), 2)

    cv2.imshow("Canny", edges)
    cv2.imshow("Contours", debug)

    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):  # ESC ou q
        break

cv2.destroyAllWindows()
cv2.waitKey(1)