import math
import numpy as np
from dataclasses import dataclass
from scipy.interpolate import interp1d

from .local_occupancy import (
    sensor_measurements,
    sensor_distances,
    sensor_pos_from_center,
    sensor_angles,
)


import numpy as np

def densify_path(path, step=20.0):
    """
    Densify a polyline path by inserting points between waypoints.

    path : list/array of shape (N, 2)  (e.g. [(x0,y0), (x1,y1), ...])
    step : desired spacing between points (same unit as path coordinates)

    Returns
    -------
    dense_path : np.ndarray of shape (M, 2)
    """
    path_arr = np.asarray(path, dtype=float)
    if path_arr.shape[0] < 2:
        return path_arr

    dense_segments = []

    for i in range(len(path_arr) - 1):
        p0 = path_arr[i]
        p1 = path_arr[i + 1]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-9:
            continue

        # number of points for this segment (>= 2)
        n_pts = max(2, int(seg_len / step) + 1)

        # interpolate from p0 to just before p1
        t = np.linspace(0.0, 1.0, n_pts, endpoint=False)
        pts = p0[None, :] + t[:, None] * seg_vec[None, :]
        dense_segments.append(pts)

    # add the last original waypoint
    dense_segments.append(path_arr[-1][None, :])

    dense_path = np.vstack(dense_segments)
    return dense_path

def wrap_to_pi(angle: float) -> float:
    """Wrap any angle (rad) to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def sensor_val_to_cm_dist(val: int) -> float:
    """Convert a Thymio prox value to a distance in cm using your calibration."""
    if val == 0:
        return np.inf
    f = interp1d(sensor_measurements, sensor_distances)
    return float(f(val))

# def obstacles_pos_from_sensor_vals(sensor_vals):
#     """
#     Convert Thymio's horizontal prox sensor values to obstacle positions
#     in ROBOT frame, in cm.

#     Robot frame:
#       x = left (+), right (-)
#       y = forward (+)
#     """
#     dists_cm = [sensor_val_to_cm_dist(v) for v in sensor_vals]

#     dx = [d * math.cos(a) for d, a in zip(dists_cm, sensor_angles)]
#     dy = [d * math.sin(a) for d, a in zip(dists_cm, sensor_angles)]

#     obstacles = []
#     for (sx, sy), ddx, ddy in zip(sensor_pos_from_center, dx, dy):
#         obstacles.append([sx + ddx, sy + ddy])

#     return np.array(obstacles, dtype=float)  # (Nsensors, 2) in cm


def obstacles_pos_from_sensor_vals(sensor_vals):
    """
    Convert Thymio's horizontal prox values to obstacle positions
    in the ROBOT frame, in centimeters.

    Robot frame:
      x_left  > 0  -> obstacle is on the LEFT
      x_left  < 0  -> obstacle is on the RIGHT
      y_forward > 0 -> obstacle is in FRONT

    Note: sensor_pos_from_center is defined with axes
    x_left ( + to robot's left ) and y_forward ( + to the front ).
    """
    dists_cm = [sensor_val_to_cm_dist(v) for v in sensor_vals]

    obstacles = []
    for (sx_left, sy_forward), d, a in zip(sensor_pos_from_center, dists_cm, sensor_angles):
        if not np.isfinite(d):
            continue

        # Original frame: x_left, y_forward
        obs_left    = sx_left    + d * math.cos(a)
        obs_forward = sy_forward + d * math.sin(a)

        x_left    = obs_left
        y_forward = obs_forward

        obstacles.append([x_left, y_forward])

    return np.array(obstacles, dtype=float)


# -------------------------------
# Occupancy grid with inflation
# -------------------------------

class LocalOccupancyGrid:
    """
    Robot-centred occupancy grid.
    x: left (+), right (-)
    y: forward (+), back (-)
    """

    def __init__(
        self,
        size_m: float = 0.6,
        resolution_m: float = 0.01,
        hit_inc: float = 0.4,
        decay: float = 0.95,
        robot_radius_m: float = 0.085,
    ):
        self.size_m = size_m
        self.resolution = resolution_m
        self.hit_inc = hit_inc
        self.decay = decay
        self.robot_radius_m = robot_radius_m

        self.half_size = size_m / 2.0
        self.n_cells = int(size_m / resolution_m)
        self.grid = np.zeros((self.n_cells, self.n_cells), dtype=float)

    def _point_to_index(self, x_m, y_m):
        if abs(x_m) > self.half_size or abs(y_m) > self.half_size:
            return None
        ix = int((x_m + self.half_size) / self.resolution)
        iy = int((y_m + self.half_size) / self.resolution)
        ix = min(max(ix, 0), self.n_cells - 1)
        iy = min(max(iy, 0), self.n_cells - 1)
        return iy, ix

    def _index_to_point(self, iy, ix):
        x_m = (ix + 0.5) * self.resolution - self.half_size
        y_m = (iy + 0.5) * self.resolution - self.half_size
        return x_m, y_m

    def update_from_sensor_vals(self, sensor_vals):
        """
        Update grid from Thymio horizontal proximity readings.
        Obstacles are inflated by robot radius.
        """
        self.grid *= self.decay
        obs_cm = obstacles_pos_from_sensor_vals(sensor_vals)

        inflate_cells = max(1, int(self.robot_radius_m / self.resolution))

        for ox_cm, oy_cm in obs_cm:
            if not np.isfinite(ox_cm) or not np.isfinite(oy_cm):
                continue

            x_m = ox_cm / 100.0
            y_m = oy_cm / 100.0

            idx = self._point_to_index(x_m, y_m)
            if idx is None:
                continue

            iy, ix = idx

            # Inflate in a disk around (iy, ix)
            for dy in range(-inflate_cells, inflate_cells + 1):
                for dx in range(-inflate_cells, inflate_cells + 1):
                    jy = iy + dy
                    jx = ix + dx
                    if 0 <= jy < self.n_cells and 0 <= jx < self.n_cells:
                        if dx * dx + dy * dy <= inflate_cells * inflate_cells:
                            self.grid[jy, jx] = min(
                                1.0, self.grid[jy, jx] + self.hit_inc
                            )


# -------------------------------
# Local navigation
# -------------------------------

@dataclass
class LocalNavConfig:
    lookahead_dist_m: float = 0.15
    max_lookahead_points: int = 30
    occ_threshold: float = 0.3 
    rep_influence_radius: float = 0.8
    k_att: float = 1.0
    k_rep: float = 0.6
    virt_goal_dist_m: float = 0.12
    #k_lat_weight=2.0


class LocalNavigator:
    def __init__(self, grid: LocalOccupancyGrid, cfg: LocalNavConfig):
        self.grid = grid
        self.cfg = cfg
        self.path_idx = 0

    @staticmethod
    def world_to_robot(pose, point_world):
        x_r, y_r, theta = pose
        px, py = point_world
        dx = px - x_r
        dy = py - y_r
        c = math.cos(theta)
        s = math.sin(theta)
        y_forward = c * dx + s * dy
        x_left = -s * dx + c * dy
        return np.array([x_left, y_forward], dtype=float)

    @staticmethod
    def robot_to_world(pose, point_robot):
        x_r, y_r, theta = pose
        x_left, y_forward = point_robot
        c = math.cos(theta)
        s = math.sin(theta)
        dx_w = -s * x_left + c * y_forward
        dy_w = c * x_left + s * y_forward
        return np.array([x_r + dx_w, y_r + dy_w], dtype=float)

    def _find_lookahead_point(self, pose, path):
        x, y, theta = pose
        if path is None or path.shape[0] == 0:
            return None, None
        
        N = path.shape[0]
        # Clamp path_idx within [0, N-1]
        if self.path_idx >= N:
            self.path_idx = N - 1
        if self.path_idx < 0:
            self.path_idx = 0

        start_idx = self.path_idx
        end_idx = min(N, start_idx + self.cfg.max_lookahead_points)
        segment = path[start_idx:end_idx]
           
         # If still empty, just say "we're at the last point"
        if segment.size == 0:
            return path[-1], N - 1
    
        dists2 = (segment[:, 0] - x) ** 2 + (segment[:, 1] - y) ** 2
        k_local = int(np.argmin(dists2))
        k_global = start_idx + k_local
        self.path_idx = k_global

        target_idx = k_global
        accum = 0.0
        while (
            target_idx + 1 < N
            and accum < self.cfg.lookahead_dist_m
            and target_idx - k_global < self.cfg.max_lookahead_points
        ):
            p0 = path[target_idx]
            p1 = path[target_idx + 1]
            step = float(np.linalg.norm(p1 - p0))
            accum += step
            target_idx += 1

        LA_point = path[target_idx]
        return LA_point, k_global

 

    def _repulsive_vector(self):
            """
            Repulsive force in ROBOT frame [x_left, y_forward].

            - Only from cells in front (y_forward > 0)
            - Weighted by occupancy
            - BOUNDED in norm so it can't completely dominate attraction
            """
            F_rep = np.zeros(2, dtype=float)
            R = self.cfg.rep_influence_radius
            eps = 1e-3

            # Cells with occupancy above threshold contribute
            occ_idxs = np.argwhere(self.grid.grid > self.cfg.occ_threshold)

            for iy, ix in occ_idxs:
                occ_val = float(self.grid.grid[iy, ix])  # in [0,1]
                x_m, y_m = self.grid._index_to_point(iy, ix)
                d = math.sqrt(x_m * x_m + y_m * y_m)

                # ignore behind robot
                if y_m <= 0.0:
                    continue
                if d < eps or d > R:
                    continue

                lat = -x_m / d
                back = -y_m / d

                dir_vec = np.array([lat, 0.4 * back], dtype=float)

                n = np.linalg.norm(dir_vec)
                if n < 1e-6:
                    continue
                dir_vec /= n

                # use a smooth, bounded-ish influence: stronger when closer,
                # but not blowing up to infinity
                # s = 1 at d=0, 0 at d>=R
                s = max(0.0, (R - d) / R)
                # square it so far obstacles are weaker
                s = s * s

                mag = self.cfg.k_rep * occ_val * s

                F_rep += mag * dir_vec

            # ---- FINAL SATURATION on F_rep magnitude ----
            # we don't want repulsion to be 100x the attraction,
            # otherwise the robot can get thrown far off the path.
            max_F_rep = 2.0 * self.cfg.k_att  # repulsion can be up to ~2.0x attraction
            
            norm_rep = np.linalg.norm(F_rep)
            if norm_rep > max_F_rep:
                F_rep = F_rep * (max_F_rep / norm_rep)

            return F_rep

    

    def compute_virtual_goal(self, pose, path, sensor_vals=None):
        if path is None or len(path) == 0:
            return None, None, None, None

        path_arr = np.asarray(path, dtype=float)
        LA_world, _ = self._find_lookahead_point(pose, path_arr)
        if LA_world is None:
            return None, None, None, None

        p_LA_robot = self.world_to_robot(pose, LA_world)

        F_att = p_LA_robot.astype(float)
        norm_att = np.linalg.norm(F_att)
        if norm_att > 1e-6:
            F_att = (self.cfg.k_att / norm_att) * F_att

        F_rep = self._repulsive_vector()
       
        if sensor_vals is None or max(sensor_vals) < 300:
            # no significant obstacle: go straight to lookahead
            return LA_world, LA_world, F_att, np.zeros(2)
        #if sensor_vals is not None and max(sensor_vals) < 300:
         #   F_rep *= 0.2

        F_tot = F_att + F_rep

        if sensor_vals is not None:
            left_side = max(sensor_vals[0], sensor_vals[1])
            right_side = max(sensor_vals[3], sensor_vals[4])

            if left_side > 1200:
                # obstacle gauche -> pousse l’angle à droite
                F_tot[0] -= 0.25

            if right_side > 1200:
                # obstacle droite -> pousse l’angle à gauche
                F_tot[0] += 0.25


        # Si obstacle FRONT CENTER très fort
        if sensor_vals is not None and sensor_vals[2] > 1500:
            F_tot[1] = min(F_tot[1], 0.05)
            
        norm_tot = np.linalg.norm(F_tot)#Normalisatiopn 
        if norm_tot < 1e-6:
            F_tot = np.array([0.0, 1.0], dtype=float)
            norm_tot = 1.0

        dir_robot = F_tot / norm_tot
        virt_robot = self.cfg.virt_goal_dist_m * dir_robot
        virt_world = self.robot_to_world(pose, virt_robot)
        
        return virt_world, LA_world, F_att, F_rep


# -------------------------------
# Go-to-goal controller
# -------------------------------

@dataclass
class GoToGoalGains:
    Kv: float = 2.0
    Komega: float = 3.0
    v_max: float = 0.18
    w_max: float = 2.0

@dataclass
class ThymioKinematics:
    wheel_radius: float = 0.022
    half_axle: float = 0.0475
    motor_gain: float = 10.0
    motor_max: int = 300

class GoToGoalController:
    def __init__(self, gains: GoToGoalGains, kin: ThymioKinematics):
        self.g = gains
        self.kin = kin

    def compute_unicycle_cmd(self, pose, goal_xy):
        x, y, theta = pose
        xg, yg = goal_xy
        dx = xg - x
        dy = yg - y

        rho = math.hypot(dx, dy)

        y_forward = math.cos(theta) * dx + math.sin(theta) * dy
        x_left = math.sin(theta) * dx - math.cos(theta) * dy
        print("x:",x_left,"y:", y_forward)
        alpha = math.atan2(x_left, y_forward)
        print("alpha", alpha)
        alpha = wrap_to_pi(alpha)
        print("wraped alpha", alpha)

        alpha_dead = 5.0 * math.pi / 180.0
        if abs(alpha) < alpha_dead:
            alpha = 0.0
        print("dead_alpha", alpha)
        """           
        alpha_dead = 30.0 * math.pi / 180.0  # ±5°

        # Deadzone autour de +π
        if abs(alpha - math.pi) < alpha_dead:
            alpha = math.pi

        # Deadzone autour de -π
        elif abs(alpha + math.pi) < alpha_dead:
            alpha = -math.pi
        """

        v = self.g.Kv * rho
        w = self.g.Komega * alpha
        print("v,w before clamp:", v,"  ",w)
        v = max(-self.g.v_max, min(self.g.v_max, v))
        w = max(-self.g.w_max, min(self.g.w_max, w))
        print("v,w after clamp:", v,"  ",w)

        return v, w, rho, alpha

    def unicycle_to_wheels(self, v, w):
        r = self.kin.wheel_radius
        L = self.kin.half_axle

        # Correct mapping
        phi_L = (v / r) - (L / r) * w
        phi_R = (v / r) + (L / r) * w

        uL = int(self.kin.motor_gain * phi_L)
        uR = int(self.kin.motor_gain * phi_R)

        uL = max(-self.kin.motor_max, min(self.kin.motor_max, uL))
        uR = max(-self.kin.motor_max, min(self.kin.motor_max, uR))

        return uL, uR

    def compute_motor_commands(self, pose, goal_xy, tol=0.02, stop_if_close=True):
        v, w, rho, alpha = self.compute_unicycle_cmd(pose, goal_xy)
        if stop_if_close and rho < tol:
            return 0, 0, {"rho": rho, "alpha": alpha, "stopped": True}
        uL, uR = self.unicycle_to_wheels(v, w)
        return uL, uR, {"rho": rho, "alpha": alpha, "stopped": False}

grid = LocalOccupancyGrid(
    size_m=0.8,
    resolution_m=0.01,
    hit_inc=0.4,   # was 0.8 – each hit = smaller bump
    decay=0.85,    # was 0.9 – forget faster
    robot_radius_m=0.12,
)

cfg = LocalNavConfig(
    lookahead_dist_m=0.15,
    max_lookahead_points=30,
    occ_threshold=0.3,
    rep_influence_radius=0.8,
    k_att=1.2,
    k_rep=0.85,
    virt_goal_dist_m=0.10,
    #k_lat_weight=2.0,   # try 2–3
)

navigator = LocalNavigator(grid, cfg)

gains = GoToGoalGains(
    Kv=2.0,
    Komega=10.0,
    v_max=0.3,
    w_max=7.0,
)
kin = ThymioKinematics()
g2g = GoToGoalController(gains, kin)
