import sys
import os

# --- pour pouvoir importer computer_vision.vision facilement ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import numpy as np
from shapely.geometry import Polygon
import pyvisgraph as vg
#import networkx as nx
import math

from computer_vision.vision import Vision, initialisation


def inflate_obstacles(obstacles, robot_radius):

    inflated = []
    for poly in obstacles:
        P = Polygon(poly)
        P_inflated = P.buffer(robot_radius,join_style=2)

        inflated_poly = list(P_inflated.exterior.coords)[:-1]
        inflated.append(inflated_poly)
    return inflated



def compute_visibility_path(inflated_obstacles, start, goal):

    # Convertir les obstacles en polygones pyvisgraph
    polygons = []
    for poly in inflated_obstacles:
        pts = [vg.Point(float(x), float(y)) for (x, y) in poly]
        polygons.append(pts)

    # Construire le graphe de visibilité
    g = vg.VisGraph()
    g.build(polygons)  

    # Points de départ / arrivée
    s = vg.Point(float(start[0]), float(start[1]))
    t = vg.Point(float(goal[0]),  float(goal[1]))

    # Plus court chemin
    path_pts = g.shortest_path(s, t)

    # Repasser en tuples (x, y)
    path = [(p.x, p.y) for p in path_pts]
    return path,g



if __name__ == "__main__":
    # Vision + image
    vision = Vision()
    img = vision.load("table4.jpg")  

    vision.params = initialisation(img)

    # Détection des obstacles avec CES paramètres
    obstacles = vision.get_state(img)
    

    # Appliquer le buffer uniquement maintenant
    robot_radius = 40      # à ajuster en pixels
    inflated = inflate_obstacles(obstacles, robot_radius)

    out = img.copy()

    # obstacles originaux en VERT
    for poly in obstacles:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)

    # obstacles gonflés en ROUGE
    for poly in inflated:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 0, 255), 2)

    cv2.imshow("Obstacles + buffer", out)
    cv2.imwrite("inflate_test.png", out)

    # définir start / goal (temporaire)
    start = (250, 250)
    goal  = (1200, 500)

    # calcul du shortest_path
    path,g = compute_visibility_path(inflated, start, goal)
    print("Chemin trouvé :")
    for p in path:
        print("  ", p)

    path_img = out.copy()

    # dessiner start / goal
    cv2.circle(path_img, (int(start[0]), int(start[1])), 8, (255, 0, 0), -1)
    cv2.circle(path_img, (int(goal[0]),  int(goal[1])),  8, (0, 0, 255), -1)

    # --- dessiner toutes les edges visibles ---
    edges = g.visgraph.get_edges()

    for edge in edges:
        x1, y1 = int(edge.p1.x), int(edge.p1.y)
        x2, y2 = int(edge.p2.x), int(edge.p2.y)

        cv2.line(path_img, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # dessiner les segments du path
    for i in range(len(path) - 1):
        x1, y1 = map(int, path[i])
        x2, y2 = map(int, path[i+1])
        cv2.line(path_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.imshow("Path", path_img)
    cv2.imwrite("path_result.png", path_img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
