import cv2
import numpy as np
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union
import pyvisgraph as vg


class GlobalNavigator:
    """
    Static class for global path planning from obstacles, start, and goal.
    """

    @staticmethod
    def inflate_obstacles(obstacles, robot_radius_px, buffer_resolution=4, simplify_tolerance=15.0):
        """
        Inflate obstacles by the robot radius with reduced vertices for faster path planning.

        Args:
            obstacles: list of polygons (list of (x, y) tuples)
            robot_radius_px: radius of the robot in pixels
            buffer_resolution: number of points to approximate a quarter circle (lower = fewer points)
            simplify_tolerance: Shapely simplify tolerance in pixels

        Returns:
            List of inflated polygons (list of (x, y) tuples)
        """
        inflated = []
        for poly in obstacles:
            P = Polygon(poly)

            # Inflate with lower resolution to reduce vertices
            P_inflated = P.buffer(robot_radius_px, resolution=buffer_resolution, join_style=2)

            # Optional: simplify polygon to further reduce vertices
            P_inflated = P_inflated.simplify(simplify_tolerance, preserve_topology=True)

            # Convert back to list of coordinates
            if P_inflated.is_empty:
                continue
            if P_inflated.geom_type == "Polygon":
                inflated.append(list(P_inflated.exterior.coords)[:-1])
            elif P_inflated.geom_type == "MultiPolygon":
                for g in P_inflated.geoms:
                    inflated.append(list(g.exterior.coords)[:-1])

        return inflated

    @staticmethod
    def merge_inflated_obstacles(inflated_obstacles):
        polys = [Polygon(p) for p in inflated_obstacles]
        merged = unary_union(polys)
        merged_list = []
        if merged.geom_type == 'Polygon':
            merged_list.append(list(merged.exterior.coords)[:-1])
        elif merged.geom_type == 'MultiPolygon':
            for geom in merged.geoms:
                merged_list.append(list(geom.exterior.coords)[:-1])
        return merged_list
    
    @staticmethod    
    def clip_obstacles_to_image(inflated_obstacles, border):

        clipped = []

        for poly in inflated_obstacles:
            P = Polygon(poly)

            inter = P.intersection(border)

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

    @staticmethod
    def outside_obstacle(obstacles, coord, margin=None):
        """
        Check if a robot at coord is outside all obstacles.
        If robot_radius_px is given, approximate the robot with a square (AABB).
        """
        if margin is not None:
            # Create axis-aligned bounding box around the robot
            x, y = coord
            half_r = margin
            pose = box(x - half_r, y - half_r, x + half_r, y + half_r)
        else:
            pose = Point(coord[0], coord[1])

        for poly in obstacles:
            P = Polygon(poly)
            if P.intersects(pose):
                return False
        return True

    
    @staticmethod
    def compute_visibility_path(merged_inflated, start, goal,border):

        # Convertir les obstacles en polygones pyvisgraph
        polygons = []
        for poly in merged_inflated:
            pts = [vg.Point(float(x), float(y)) for (x, y) in poly]
            polygons.append(pts)

        # Construire le graphe de visibilité
        g = vg.VisGraph()
        g.build(polygons) 

        VG = g.visgraph
        edges_to_remove = set()

        for edge in list(VG.get_edges()):
            seg = LineString([(edge.p1.x, edge.p1.y),
                            (edge.p2.x, edge.p2.y)])

            # On veut que le segment soit entièrement DANS la zone navigable
            # (on tolère un léger flottement numérique avec buffer(-eps))
            if not seg.within(border):
                edges_to_remove.add(edge)

        for e in edges_to_remove:
            VG.graph[e.p1].discard(e)
            VG.graph[e.p2].discard(e)
            VG.edges.discard(e)

        # Points de départ / arrivée
        s = vg.Point(float(start[0]), float(start[1]))
        t = vg.Point(float(goal[0]),  float(goal[1]))

        try:
            path_pts = g.shortest_path(s, t)
        except KeyError as e:
            print(f"[GLOBAL] No possible path (KeyError in shortest_path) : {e}")
            return None, g

        # Repasser en tuples (x, y)
        path = [(p.x, p.y) for p in path_pts]
        return path,g

    @staticmethod
    def path_is_collision_free(path, obstacles):
        polys = [Polygon(poly) for poly in obstacles]
        for i in range(len(path) - 1):
            seg = LineString([path[i], path[i + 1]])
            for P in polys:
                if seg.crosses(P) or seg.within(P):
                    return False
        return True

    @staticmethod
    def plan_path(obstacles, start, goal, robot_radius_px=0, img_width=None, img_height=None, debug_img=None):
        """
        Compute a global collision-free path from a start to a goal.

        Args:
            obstacles: list of polygons (each polygon is a list of (x, y) tuples)
            start: (x, y) tuple of robot position
            goal: (x, y) tuple of goal position
            robot_radius_px: optional radius for obstacle inflation
            img_width, img_height: optional, used for clipping obstacles
            debug_img: optional image to draw path and obstacles

        Returns:
            path: list of (x, y) tuples, or None if no valid path
        """

        border = box(
            robot_radius_px,
            robot_radius_px,
            img_width - robot_radius_px,
            img_height - robot_radius_px
        )

        # 1) Inflate obstacles
        inflated = GlobalNavigator.inflate_obstacles(obstacles, robot_radius_px)

        # 3) Merge overlapping inflated obstacles
        merged_inflated = GlobalNavigator.merge_inflated_obstacles(inflated)

        merged_inflated_clip = GlobalNavigator.clip_obstacles_to_image(merged_inflated, border)


        # 4) Validate start and goal positions
        if not GlobalNavigator.outside_obstacle(merged_inflated, start, robot_radius_px/2):
            print("[GLOBAL] Start position invalid:", start)
            return None
        if not GlobalNavigator.outside_obstacle(merged_inflated, goal, robot_radius_px/2):
            print("[GLOBAL] Goal position invalid:", goal)
            return None

        # 5) Compute shortest path using visibility graph
        path, g = GlobalNavigator.compute_visibility_path(merged_inflated_clip, start, goal,border)
        if not GlobalNavigator.path_is_collision_free(path, obstacles) and path != None:
            print("[GLOBAL] No collision-free path found.")
            return None

        # 6) Optional debug drawing
        if debug_img is not None:
            debug_copy = debug_img.copy()
            for poly in obstacles:
                cv2.polylines(debug_copy, [np.array(poly, np.int32)], True, (0, 255, 0), 2)
            for poly in merged_inflated_clip:
                cv2.polylines(debug_copy, [np.array(poly, np.int32)], True, (0, 0, 255), 2)
            for edge in g.visgraph.get_edges():
                cv2.line(debug_copy, (int(edge.p1.x), int(edge.p1.y)),
                         (int(edge.p2.x), int(edge.p2.y)), (255, 255, 0), 1)
            cv2.circle(debug_copy, start, 8, (255, 0, 0), -1)
            cv2.circle(debug_copy, goal, 8, (0, 0, 255), -1)
            for i in range(len(path) - 1):
                cv2.line(debug_copy, tuple(map(int, path[i])), tuple(map(int, path[i + 1])), (255, 0, 0), 3)
            
            return path, debug_copy

        print("[GLOBAL] Path found with", len(path), "points.")
        return path, None