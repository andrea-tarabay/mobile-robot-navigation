import math
import time
import numpy as np
from dataclasses import dataclass
from scipy.interpolate import interp1d

# If this is inside a package, keep the leading dot. If standalone, remove the dot.
from local_navigation.local_occupancy import (
    sensor_measurements,
    sensor_distances,
    sensor_pos_from_center,
    sensor_angles,
)

#hello
# -------------------------------
# Path densification
# -------------------------------

def densify_path(path, step=0.02):
    """
    Densify a polyline path by inserting points between waypoints.

    path : (N, 2) array-like, in meters
    step : desired spacing between points [m]

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

        n_pts = max(2, int(seg_len / step) + 1)
        t = np.linspace(0.0, 1.0, n_pts, endpoint=False)
        pts = p0[None, :] + t[:, None] * seg_vec[None, :]
        dense_segments.append(pts)

    dense_segments.append(path_arr[-1][None, :])
    dense_path = np.vstack(dense_segments)
    return dense_path


# -------------------------------
# Basic utilities
# -------------------------------

def wrap_to_pi(angle: float) -> float:
    """Wrap any angle (rad) to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def sensor_val_to_cm_dist(val: int) -> float:
    """Convert a Thymio prox value to a distance in cm using your calibration."""
    if val == 0:
        return np.inf
    f = interp1d(sensor_measurements, sensor_distances)
    return float(f(val))


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
    lookahead_dist_m: float = 0.06
    max_lookahead_points: int = 30
    occ_threshold: float = 0.3
    rep_influence_radius: float = 0.05
    k_att: float = 1.0
    k_rep: float = 0.6
    virt_goal_dist_m: float = 0.12


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
        if self.path_idx >= N:
            self.path_idx = N - 1
        if self.path_idx < 0:
            self.path_idx = 0

        start_idx = self.path_idx
        end_idx = min(N, start_idx + self.cfg.max_lookahead_points)
        segment = path[start_idx:end_idx]

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

        occ_idxs = np.argwhere(self.grid.grid > self.cfg.occ_threshold)

        for iy, ix in occ_idxs:
            occ_val = float(self.grid.grid[iy, ix])  # in [0,1]
            x_m, y_m = self.grid._index_to_point(iy, ix)
            d = math.sqrt(x_m * x_m + y_m * y_m)

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

            s = max(0.0, (R - d) / R)
            s = s * s

            mag = self.cfg.k_rep * occ_val * s
            F_rep += mag * dir_vec

        max_F_rep = 2.0 * self.cfg.k_att
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
        print("print attractive force",F_att)
        F_rep = self._repulsive_vector()

        # If no significant obstacles, ignore repulsion but STILL use a virtual goal
        if sensor_vals is None or max(sensor_vals) < 300:
            F_rep = np.zeros(2)

        F_tot = F_att + F_rep

        # (keep your heuristics on F_tot)
        if sensor_vals is not None:
            left_side  = max(sensor_vals[0], sensor_vals[1])
            right_side = max(sensor_vals[3], sensor_vals[4])

            if left_side > 1200:
                F_tot[0] -= 0.25
            if right_side > 1200:
                F_tot[0] += 0.25

        if sensor_vals is not None and sensor_vals[2] > 1500:
            F_tot[1] = min(F_tot[1], 0.05)

        norm_tot = np.linalg.norm(F_tot)
        if norm_tot < 1e-6:
            F_tot = np.array([0.0, 1.0], dtype=float)
            norm_tot = 1.0

        dir_robot = F_tot / norm_tot
        virt_robot = self.cfg.virt_goal_dist_m * dir_robot
        virt_world = self.robot_to_world(pose, virt_robot)

        return virt_world, LA_world, F_att, F_rep


# -------------------------------
# Heuristics (side + center)
# -------------------------------

def apply_side_heuristic(sensor_vals, w, min_turn=1.5, side_thr=1200):
    """
    Use outer front sensors to enforce TURN DIRECTION and minimum |w|.

    Thymio prox.horizontal:
        0: front left
        1: front left-center
        2: front center
        3: front right-center
        4: front right
        5: rear left
        6: rear right

    - If right side is stronger -> force LEFT turn (w > 0)
    - If left  side is stronger -> force RIGHT turn (w < 0)
    """
    if sensor_vals is None or len(sensor_vals) < 5:
        return w

    s = sensor_vals
    left_side = max(s[0], s[1])
    right_side = max(s[3], s[4])

    if left_side < side_thr and right_side < side_thr:
        return w

    if right_side > left_side and right_side >= side_thr:
        # obstacle on RIGHT -> turn LEFT
        if w <= 0:
            w = min_turn
        else:
            w = max(w, min_turn)
    elif left_side > right_side and left_side >= side_thr:
        # obstacle on LEFT -> turn RIGHT
        if w >= 0:
            w = -min_turn
        else:
            w = min(w, -min_turn)

    return w


def apply_center_heuristic(sensor_vals, v, w, min_turn=1.8, center_thr=1500):
    """
    Special handling when the FRONT-CENTER sensor (index 2) sees something.

    - Strong turn left/right depending on which side also sees the obstacle more.
    - Slightly reduce v so the robot turns harder and doesn't hit the obstacle.
    """
    if sensor_vals is None or len(sensor_vals) < 3:
        return v, w

    s = sensor_vals
    center_val = s[2]

    if center_val < center_thr:
        return v, w

    left_side = max(s[0], s[1])
    right_side = max(s[3], s[4])

    if left_side < center_thr and right_side < center_thr:
        # Both sides small -> just enforce min |w| with current sign
        if w >= 0:
            w = max(w, min_turn)
        else:
            w = min(w, -min_turn)
    else:
        # Stronger on the right -> turn left
        if right_side >= left_side:
            if w <= 0:
                w = min_turn
            else:
                w = max(w, min_turn)
        # Stronger on the left -> turn right
        else:
            if w >= 0:
                w = -min_turn
            else:
                w = min(w, -min_turn)

    # reduce forward speed when something is dead ahead
    v = min(v, 0.05)
    return v, w


# -------------------------------
# Go-to-goal controller
# -------------------------------

@dataclass
class GoToGoalGains:
    Kv: float = 2.0
    Komega: float = 3.0      # smaller than 4.0 now that we have D
    Ki_omega: float = 0.0    # you can try 0.1 later
    Kd_omega: float = 0.6
    int_alpha_max: float = 0.5
    Ki_v: float = 0
    v_max: float = 0.25
    w_max: float = 1.5
   


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
         # PID state for angle
        self.prev_alpha = 0.0
        self.int_alpha = 0.0
        self.int_rho = 0.0
        self.prev_time = None

    # def compute_unicycle_cmd(self, pose, goal_xy):
    #     x, y, theta = pose
    #     xg, yg = goal_xy
    #     dx = xg - x
    #     dy = yg - y

    #     rho = math.hypot(dx, dy)

    #     y_forward = math.cos(theta) * dx + math.sin(theta) * dy
    #     x_left = math.sin(theta) * dx - math.cos(theta) * dy
    #     print("x:", x_left, "y:", y_forward)
    #     alpha = math.atan2(x_left, y_forward)
    #     print("alpha", alpha)
    #     alpha = wrap_to_pi(alpha)
    #     print("wraped alpha", alpha)

    #     alpha_dead = 5.0 * math.pi / 180.0
    #     if abs(alpha) < alpha_dead:
    #         alpha = 0.0
    #     print("dead_alpha", alpha)

    #     v = self.g.Kv * rho
    #     w = self.g.Komega * alpha
    #     print("v,w before clamp:", v, "  ", w)
    #     v = max(-self.g.v_max, min(self.g.v_max, v))
    #     w = max(-self.g.w_max, min(self.g.w_max, w))
    #     print("v,w after clamp:", v, "  ", w)

    #     return v, w, rho, alpha

    def compute_unicycle_cmd(self, pose, goal_xy, dt=None):
        x, y, theta = pose
        xg, yg = goal_xy
        dx = xg - x
        dy = yg - y

        rho = math.hypot(dx, dy)
        print("x:", x,"xg:",xg, "y:", y,"yg:", yg, "dx", dx, "dy", dy, "rho", rho)

        # angle of the goal in robot frame
        y_forward = math.cos(theta) * dx + math.sin(theta) * dy
        x_left = math.sin(theta) * dx - math.cos(theta) * dy
        print("x:", x_left, "y:", y_forward)

        alpha = math.atan2(x_left, y_forward)
        print("alpha", alpha)
        alpha = wrap_to_pi(alpha)
        print("wraped alpha", alpha)

        # deadzone on angle
        alpha_dead = 5.0 * math.pi / 180.0
        if abs(alpha) < alpha_dead:
            alpha = 0.0
        print("dead_alpha", alpha)

        # ---------------------------
        #   PID ON ANGLE (alpha)
        # ---------------------------

        # 1) Determine dt
        if dt is None:
            now = time.time()
            if self.prev_time is None:
                dt = 0.1  # reasonable default first step
            else:
                dt = now - self.prev_time
                # clamp dt so spikes don't explode derivative & integral
                dt = max(0.01, min(dt, 0.2))
            self.prev_time = now
        else:
            # if user passes dt, we just clamp it
            dt = max(0.001, float(dt))

        # 2) Integral term (with anti-windup)
        self.int_alpha += alpha * dt
        # clamp integral to avoid windup
        self.int_alpha = max(-self.g.int_alpha_max,
                             min(self.g.int_alpha_max, self.int_alpha))

        # 3) Derivative term
        alpha_dot = (alpha - self.prev_alpha) / dt
        self.prev_alpha = alpha

        # 2) Integral term (with anti-windup)
        self.int_rho += rho * dt



        # 4) PID output for angular velocity
        Kp = self.g.Komega
        Ki = self.g.Ki_omega
        Kd = self.g.Kd_omega

        w = Kp * alpha + Ki * self.int_alpha + Kd * alpha_dot

        Kiv = self.g.Ki_v

        # 5) Linear velocity (still simple P on rho)
        v = self.g.Kv * rho + Kiv * self.int_rho

        print(f"PID terms: P={Kp*alpha:.3f}, I={Ki*self.int_alpha:.3f}, "
              f"D={Kd*alpha_dot:.3f}, w_raw={w:.3f}")
        print("v,w before clamp:", v, "  ", w)

        # Clamp
        v = max(-self.g.v_max, min(self.g.v_max, v))
        w = max(-self.g.w_max, min(self.g.w_max, w))
        print("v,w after clamp:", v, "  ", w)

        return v, w, rho, alpha


    def unicycle_to_wheels(self, v, w):
        r = self.kin.wheel_radius
        L = self.kin.half_axle

        phi_L = (v / r) - (L / r) * w
        phi_R = (v / r) + (L / r) * w

        uL = int(self.kin.motor_gain * phi_L)
        uR = int(self.kin.motor_gain * phi_R)

        uL = max(-self.kin.motor_max, min(self.kin.motor_max, uL))
        uR = max(-self.kin.motor_max, min(self.kin.motor_max, uR))

        return uL, uR

    def compute_motor_commands(self, pose, goal_xy, sensor_vals=None,
                               tol=0.02, stop_if_close=True, dt=0.1):
        """
        Same as before, but now:
        - Uses side + center heuristics (if sensor_vals provided).
        - Returns v,w in the info dict (for odometry).
        """
        v, w, rho, alpha = self.compute_unicycle_cmd(pose, goal_xy, dt=dt)
        stopped = False

        if stop_if_close and rho < tol:
            v = 0.0
            w = 0.0
            stopped = True
        else:
            if sensor_vals is not None:
                # side heuristic: choose turn direction + min |w|
                w = apply_side_heuristic(sensor_vals, w,
                                         min_turn=1.5,
                                         side_thr=1200)
                # center heuristic: strong reaction if front-center is high
                v, w = apply_center_heuristic(sensor_vals, v, w,
                                              min_turn=1.8,
                                              center_thr=1500)

            # re-clamp after heuristics
            v = max(-self.g.v_max, min(self.g.v_max, v))
            w = max(-self.g.w_max, min(self.g.w_max, w))

        uL, uR = self.unicycle_to_wheels(v, w)
        return uL, uR, {"rho": rho, "alpha": alpha,
                        "stopped": stopped, "v": v, "w": w}


# -------------------------------
# Instantiate grid, navigator, controller
# -------------------------------

grid = LocalOccupancyGrid(
    size_m=0.8,
    resolution_m=0.01,
    hit_inc=0.4,
    decay=0.85,
    robot_radius_m=0.12,
)

cfg = LocalNavConfig(
    lookahead_dist_m=0.06,
    max_lookahead_points=30,
    occ_threshold=0.3,
    rep_influence_radius=0.8,#0.05 was before
    k_att=3,
    k_rep=0.6, #0.5 was before
    virt_goal_dist_m=0.20,
)

navigator = LocalNavigator(grid, cfg)

# gains = GoToGoalGains(
#     Kv=2.0,
#     Komega=4.0,
#     v_max=0.25,
#     w_max=1.5,
# )

gains = GoToGoalGains(
    Kv=3.0,
    Komega=3.0,       # smaller than 4.0 now that we have D
    Ki_omega=0.1,     # you can try 0.1 later
    Kd_omega=0.7,
    int_alpha_max=0.6,
    Ki_v = 0,
    v_max=0.4,
    w_max=2,
)
kin = ThymioKinematics()
g2g = GoToGoalController(gains, kin)


