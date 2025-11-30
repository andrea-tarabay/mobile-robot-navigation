import sys
import os

# --- pour pouvoir importer computer_vision.vision facilement ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import numpy as np
from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import unary_union
import pyvisgraph as vg
import math

from computer_vision.vision import Vision




def inflate_obstacles(obstacles, robot_radius):

    inflated = []
    for poly in obstacles:
        P = Polygon(poly)
        P_inflated = P.buffer(robot_radius,join_style=2)

        inflated_poly = list(P_inflated.exterior.coords)[:-1]
        inflated.append(inflated_poly)
    return inflated

def clip_obstacles_to_image(inflated_obstacles, img_width, img_height, robot_radius):

    # Rectangle [0,0] -> [width, height]
    bounds = box(
        robot_radius, 
        robot_radius, 
        img_width - robot_radius, 
        img_height - robot_radius
    )

    clipped = []
    for poly in inflated_obstacles:
        P = Polygon(poly)
        inter = P.intersection(bounds)

        if inter.is_empty:
            continue

        if inter.geom_type == "Polygon":
            coords = list(inter.exterior.coords)[:-1]
            clipped.append(coords)

        elif inter.geom_type == "MultiPolygon":
            for g in inter.geoms:
                coords = list(g.exterior.coords)[:-1]
                clipped.append(coords)

    return clipped


def merge_inflated_obstacles(inflated_obstacles):

    polys = [Polygon(p) for p in inflated_obstacles]
    merged = unary_union(polys)   # peut renvoyer Polygon ou MultiPolygon

    merged_list = []

    if merged.geom_type == 'Polygon':
        coords = list(merged.exterior.coords)[:-1]
        merged_list.append(coords)
    elif merged.geom_type == 'MultiPolygon':
        for geom in merged.geoms:
            coords = list(geom.exterior.coords)[:-1]
            merged_list.append(coords)

    return merged_list

def outside_obstacle(obstacles, coord, robot_radius = None):

    outside = True
    pose = Point(float(coord[0]), float(coord[1]))

    if robot_radius != None:
        pose = pose.buffer(robot_radius)


    for poly in obstacles:
        P = Polygon(poly)
        if P.intersects(pose):
            outside = False

    return outside


def build_navigation_map(img, obstacles, robot_radius):

    # 1) Gonflage
    inflated = inflate_obstacles(obstacles, robot_radius)

    # 2) Fusion des obstacles gonflés qui se chevauchent
    merged_inflated = merge_inflated_obstacles(inflated)

    map_img = img.copy()

    # Obstacles originaux en VERT
    for poly in obstacles:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(map_img, [pts], True, (0, 255, 0), 2)

    # Obstacles gonflés fusionnés en ROUGE
    for poly in merged_inflated:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(map_img, [pts], True, (0, 0, 255), 2)

    return merged_inflated, map_img

def compute_visibility_path(merged_inflated, start, goal):

    # Convertir les obstacles en polygones pyvisgraph
    polygons = []
    for poly in merged_inflated:
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


def path_is_collision_free(path, obstacles):

    polys = [Polygon(poly) for poly in obstacles]

    # on parcourt chaque segment du chemin
    for i in range(len(path) - 1):
        p0 = path[i]
        p1 = path[i+1]
        seg = LineString([p0, p1])

        for P in polys:
            # on autorise juste le cas où le segment touche la frontière
            # mais on refuse les vraies intersections "à l'intérieur"
            if seg.crosses(P) or seg.within(P):
                return False

    return True


def draw_global_path(base_img, merged_inflated, start, goal):

    # Calcul du chemin + graphe pyvisgraph
    path, g = compute_visibility_path(merged_inflated, start, goal)

    path_img = base_img.copy()

    # 1) dessiner toutes les arêtes de visibilité en cyan
    edges = g.visgraph.get_edges()
    for edge in edges:
        x1, y1 = int(edge.p1.x), int(edge.p1.y)
        x2, y2 = int(edge.p2.x), int(edge.p2.y)
        cv2.line(path_img, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # 2) dessiner start / goal
    cv2.circle(path_img, (int(start[0]), int(start[1])), 8, (255, 0, 0), -1)
    cv2.circle(path_img, (int(goal[0]),  int(goal[1])),  8, (0, 0, 255), -1)

    # 3) dessiner le chemin en bleu
    for i in range(len(path) - 1):
        x1, y1 = map(int, path[i])
        x2, y2 = map(int, path[i+1])
        cv2.line(path_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    return path, path_img



def global_nav(vision,radius,w,h,img):
    robot_radius = radius # ici a modifier


    obstacles, robot_state = vision.process(
    img,
    show_debug=False,    # no debug display from inside Vision
    skip_blur=False,
    log_pose=False
    )

    inflated = inflate_obstacles(obstacles, radius)
    inflated = clip_obstacles_to_image(inflated, w, h,radius)

    merged_inflated = merge_inflated_obstacles(inflated)
    out = img.copy()

    # obstacles originaux en VERT
    for poly in obstacles:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)

    # obstacles gonflés en ROUGE
    for poly in merged_inflated:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 0, 255), 2)

    cv2.imshow("Obstacles + buffer", out)
    cv2.imwrite("inflate_test.png", out)

    # définir start / goal (temporaire)
    robot = vision.detect_robot(img)  # apply_warp=True par défaut -> coordonnées déjà redressées
    robot = robot_state
    if robot["found"]:
        start = robot["center"]
        if outside_obstacle(merged_inflated, start, robot_radius):
            print("[GLOBAL] Robot trouvé avec postion valide, centre =", start,
                "theta(deg) =", np.degrees(robot["theta"]))
        else:
            print("[GLOBAL] Robot trouvé mais position non valide, centre =", start,
                "theta(deg) =", np.degrees(robot["theta"]))            
    else:
        print("[GLOBAL] Robot non trouvé sur l’image de référence")

    goal  = (1650, 1030)
    if outside_obstacle(merged_inflated, goal):
            print("[GLOBAL] Goal valide, coordonnées =", goal)

            # calcul du shortest_path
            path,g = compute_visibility_path(merged_inflated, start, goal)

            if not path_is_collision_free(path, obstacles):
                print("[GLOBAL] Aucun chemin valide (chemin impossible ou collision avec un obstacle gonflé).")
            else:
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

                    cv2.line(path_img, (x1, y1), (x2, y2), (255, 255, 0), 1)

                # dessiner les segments du path
                for i in range(len(path) - 1):
                    x1, y1 = map(int, path[i])
                    x2, y2 = map(int, path[i+1])
                    cv2.line(path_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                cv2.imshow("Path", path_img)
                cv2.imwrite("path_result.png", path_img)
                return path

    else:
            print("[GLOBAL] Goal non valide, coordonnées =", goal,)






# # if _name_ == "_main_":
# #     # Vision + image
# #     vision = Vision()
# #     img = vision.load("initial_state.png")

#     """
#                 # 1) Ouvre la caméra et capture UNE image de référence
#     cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

#     if not cap.isOpened():
#         raise RuntimeError("Impossible d'ouvrir la webcam.")

#     ret, frame0 = cap.read()
#     if not ret:
#         cap.release()
#         raise RuntimeError("Impossible de capturer une frame webcam.")

#     cap.release()  # on ferme direct, l’image est stockée dans frame0

    
#     """
    


#     cv2.imwrite("initial_state.png", img)

#     vision.initialize(img)  # couleurs + blur robot + canny + polygones (et fige dans static_polys)


#     img_h, img_w = img.shape[:2]

#     # récupère les obstacles figés par initialize()
#     obstacles, robot_state = vision.process(
#         img,
#         show_debug=False,    # on ne veut pas l'image debug ici
#         skip_blur=False,
#         log_pose=False
#     )

    

#     # Appliquer le buffer uniquement maintenant
#     robot_radius = 25     # à ajuster en pixels

#     path = global_nav(obstacles,robot_radius,img_w,img_h,img)

#     while True:
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27 or key == ord('q'):
#             break

#     cv2.destroyAllWindows()
#     cv2.waitKey(1)


#     # """
#     # inflated = inflate_obstacles(obstacles, robot_radius)
#     # inflated = clip_obstacles_to_image(inflated, img_w, img_h)

#     # merged_inflated = merge_inflated_obstacles(inflated)

    # out = img.copy()

    # # obstacles originaux en VERT
    # for poly in obstacles:
    #     pts = np.array(poly, dtype=np.int32)
    #     cv2.polylines(out, [pts], True, (0, 255, 0), 2)

    # # obstacles gonflés en ROUGE
    # for poly in merged_inflated:
    #     pts = np.array(poly, dtype=np.int32)
    #     cv2.polylines(out, [pts], True, (0, 0, 255), 2)

    # cv2.imshow("Obstacles + buffer", out)
    # cv2.imwrite("inflate_test.png", out)

    # # définir start / goal (temporaire)
    # robot = vision.detect_robot(img)  # apply_warp=True par défaut -> coordonnées déjà redressées

    # robot = robot_state
    # if robot["found"]:
    #     start = robot["center"]
    #     if outside_obstacle(merged_inflated, start, robot_radius):
    #         print("[GLOBAL] Robot trouvé avec postion valide, centre =", start,
    #             "theta(deg) =", np.degrees(robot["theta"]))
    #     else:
    #         print("[GLOBAL] Robot trouvé mais position non valide, centre =", start,
    #             "theta(deg) =", np.degrees(robot["theta"]))            
    # else:
    #     print("[GLOBAL] Robot non trouvé sur l’image de référence")

    # goal  = (550, 430)
    # if outside_obstacle(merged_inflated, goal):
    #         print("[GLOBAL] Goal valide, coordonnées =", goal)

    #         # calcul du shortest_path
    #         path,g = compute_visibility_path(merged_inflated, start, goal)

    #         if not path_is_collision_free(path, obstacles):
    #             print("[GLOBAL] Aucun chemin valide (chemin impossible ou collision avec un obstacle gonflé).")
    #         else:
    #             print("Chemin trouvé :")
    #             for p in path:
    #                 print("  ", p)

    #             path_img = out.copy()

    #             # dessiner start / goal
    #             cv2.circle(path_img, (int(start[0]), int(start[1])), 8, (255, 0, 0), -1)
    #             cv2.circle(path_img, (int(goal[0]),  int(goal[1])),  8, (0, 0, 255), -1)

    #             # --- dessiner toutes les edges visibles ---
    #             edges = g.visgraph.get_edges()

    #             for edge in edges:
    #                 x1, y1 = int(edge.p1.x), int(edge.p1.y)
    #                 x2, y2 = int(edge.p2.x), int(edge.p2.y)

    #                 cv2.line(path_img, (x1, y1), (x2, y2), (255, 255, 0), 1)

    #             # dessiner les segments du path
    #             for i in range(len(path) - 1):
    #                 x1, y1 = map(int, path[i])
    #                 x2, y2 = map(int, path[i+1])
    #                 cv2.line(path_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    #             cv2.imshow("Path", path_img)
    #             cv2.imwrite("path_result.png", path_img)
    # else:
    #         print("[GLOBAL] Goal non valide, coordonnées =", goal,)

    # while True:
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == 27 or key == ord('q'):
    #         break

    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

   