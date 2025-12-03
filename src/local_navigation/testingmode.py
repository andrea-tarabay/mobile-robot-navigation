import os
import sys
import math
import asyncio
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from tdmclient import ClientAsync

# ------------------------------------------------------------------
# 0) Import your calibration from local_occupancy.py
# ------------------------------------------------------------------
# Adjust path if needed
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from local_occupancy import (
    sensor_measurements,
    sensor_distances,
    sensor_pos_from_center,
    sensor_angles,
)


# ------------------------------------------------------------------
# 1) Small helpers
# ------------------------------------------------------------------

def wrap_to_pi(angle: float) -> float:
    """Wrap any angle (rad) to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def sensor_val_to_cm_dist(val: int) -> float:
    """Convert a Thymio prox value to a distance in cm using calibration."""
    if val == 0:
        return np.inf
    f = interp1d(sensor_measurements, sensor_distances)
    return float(f(val))


def obstacles_pos_from_sensor_vals(sensor_vals):
    """
    Convert Thymio's horizontal prox sensor values to obstacle positions
    in ROBOT frame, in cm.

    Robot frame:
      x = left (+), right (-)
      y = forward (+)
    """
    dists_cm = [sensor_val_to_cm_dist(v) for v in sensor_vals]

    dx = [d * math.cos(a) for d, a in zip(dists_cm, sensor_angles)]
    dy = [d * math.sin(a) for d, a in zip(dists_cm, sensor_angles)]

    obstacles = []
    for (sx, sy), ddx, ddy in zip(sensor_pos_from_center, dx, dy):
        obstacles.append([sx + ddx, sy + ddy])  # sensor offset + ray

    return np.array(obstacles, dtype=float)  # (Nsensors, 2) in cm


# ------------------------------------------------------------------
# 2) Occupancy grid with obstacle inflation
# ------------------------------------------------------------------

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

    # ---- coord <-> index ----

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

    # ---- main update ----

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


# ------------------------------------------------------------------
# 3) Local navigation config + navigator
# ------------------------------------------------------------------

@dataclass
class LocalNavConfig:
    lookahead_dist_m: float = 0.15
    max_lookahead_points: int = 30
    occ_threshold: float = 0.3
    rep_influence_radius: float = 0.35
    k_att: float = 1.0
    k_rep: float = 1.0
    virt_goal_dist_m: float = 0.12


class LocalNavigator:
    def __init__(self, grid: LocalOccupancyGrid, cfg: LocalNavConfig):
        self.grid = grid
        self.cfg = cfg
        self.path_idx = 0

    # --- transforms ---

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

    # --- look-ahead on path ---

    def _find_lookahead_point(self, pose, path):
        x, y, theta = pose
        if path is None or path.shape[0] == 0:
            return None, None

        start_idx = self.path_idx
        end_idx = min(path.shape[0], start_idx + self.cfg.max_lookahead_points)
        segment = path[start_idx:end_idx]
        dists2 = (segment[:, 0] - x) ** 2 + (segment[:, 1] - y) ** 2
        k_local = int(np.argmin(dists2))
        k_global = start_idx + k_local
        self.path_idx = k_global

        target_idx = k_global
        accum = 0.0
        while (
            target_idx + 1 < path.shape[0]
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

    # --- repulsive vector ---

    def _repulsive_vector(self):
        F_rep = np.zeros(2, dtype=float)
        R = self.cfg.rep_influence_radius
        eps = 1e-3

        occ_idxs = np.argwhere(self.grid.grid > self.cfg.occ_threshold)

        for iy, ix in occ_idxs:
            x_m, y_m = self.grid._index_to_point(iy, ix)
            d = math.sqrt(x_m * x_m + y_m * y_m)

            # ignore behind robot
            if y_m <= 0.0:
                continue
            if d < eps or d > R:
                continue

            dir_vec = np.array([-x_m / d, -y_m / d])
            mag = self.cfg.k_rep * (1.0 / d - 1.0 / R)
            F_rep += mag * dir_vec

        return F_rep

    # --- main virtual goal computation ---

    def compute_virtual_goal(self, pose, path):
        if path is None or path.shape[0] == 0:
            return None

        LA_world, _ = self._find_lookahead_point(pose, path)
        if LA_world is None:
            return None

        p_LA_robot = self.world_to_robot(pose, LA_world)

        # Attractive
        F_att = p_LA_robot.astype(float)
        norm_att = np.linalg.norm(F_att)
        if norm_att > 1e-6:
            F_att = (self.cfg.k_att / norm_att) * F_att

        # Repulsive
        F_rep = self._repulsive_vector()

        F_tot = F_att + F_rep
        norm_tot = np.linalg.norm(F_tot)
        if norm_tot < 1e-6:
            F_tot = np.array([0.0, 1.0], dtype=float)
            norm_tot = 1.0

        dir_robot = F_tot / norm_tot
        virt_robot = self.cfg.virt_goal_dist_m * dir_robot
        virt_world = self.robot_to_world(pose, virt_robot)

        return virt_world, LA_world, F_att, F_rep


# ------------------------------------------------------------------
# 4) Go-to-goal controller for Thymio
# ------------------------------------------------------------------

@dataclass
class GoToGoalGains:
    Kv: float = 2.0
    Komega: float = 3.0
    v_max: float = 0.18
    w_max: float = 2.0


@dataclass
class ThymioKinematics:
    wheel_radius: float = 0.022   # m
    half_axle: float = 0.0475     # m
    motor_gain: float = 10.0      # rad/s -> motor units
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
        x_left = -math.sin(theta) * dx + math.cos(theta) * dy

        alpha = math.atan2(x_left, y_forward)
        alpha = wrap_to_pi(alpha)

        # small dead-zone on heading
        alpha_dead = 3.0 * math.pi / 180.0
        if abs(alpha) < alpha_dead:
            alpha = 0.0

        v = self.g.Kv * rho
        w = self.g.Komega * alpha

        v = max(-self.g.v_max, min(self.g.v_max, v))
        w = max(-self.g.w_max, min(self.g.w_max, w))

        return v, w, rho, alpha

    def unicycle_to_wheels(self, v, w):
        r = self.kin.wheel_radius
        L = self.kin.half_axle

        # CORRECT mapping
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


# ------------------------------------------------------------------
# 5) Thymio loop: follow straight path & avoid obstacle (with debug)
# ------------------------------------------------------------------

def motors(l_speed=0, r_speed=0):
    return {
        "motor.left.target": [int(l_speed)],
        "motor.right.target": [int(r_speed)],
    }


async def follow_segment_with_avoid_debug(node, client):
    # --- path: straight line along +x, 0 -> 0.8 m ---
    path_length = 0.8
    num_points = 60
    x_path = np.linspace(0.0, path_length, num_points)
    y_path = np.zeros(num_points)
    path = np.vstack((x_path, y_path)).T

    # --- local nav + controller objects ---
    grid = LocalOccupancyGrid(
        size_m=0.6,
        resolution_m=0.01,
        hit_inc=0.4,
        decay=0.95,
        robot_radius_m=0.09,
    )
    cfg = LocalNavConfig()
    navigator = LocalNavigator(grid, cfg)

    gains = GoToGoalGains()
    kin = ThymioKinematics()
    g2g = GoToGoalController(gains, kin)

    await node.wait_for_variables({"prox.horizontal"})
    pose = np.array([path[0, 0], path[0, 1], 0.0], dtype=float)
    final_goal = path[-1]

    dt = 0.10
    max_time_s = 25.0
    n_steps = int(max_time_s / dt)

    print("Starting path following with obstacle avoidance (DEBUG)...")

    for k in range(n_steps):
        sensor_vals = list(node.v.prox.horizontal)
        grid.update_from_sensor_vals(sensor_vals)

        max_occ = float(np.max(grid.grid))
        num_thr = int(np.sum(grid.grid > cfg.occ_threshold))

        virt_world, LA_world, F_att, F_rep = navigator.compute_virtual_goal(pose, path)
        uL, uR, info = g2g.compute_motor_commands(
            pose, virt_world, tol=0.02, stop_if_close=False
        )

        await node.set_variables(motors(uL, uR))

        # integrate pose (open-loop)
        r = kin.wheel_radius
        L = kin.half_axle
        phi_L = uL / kin.motor_gain
        phi_R = uR / kin.motor_gain
        v = r * (phi_R + phi_L) / 2.0
        w = r * (phi_R - phi_L) / (2.0 * L)

        pose[0] += v * math.cos(pose[2]) * dt
        pose[1] += v * math.sin(pose[2]) * dt
        pose[2] = wrap_to_pi(pose[2] + w * dt)

        dx = final_goal[0] - pose[0]
        dy = final_goal[1] - pose[1]
        dist_to_goal = math.hypot(dx, dy)

        if k % 5 == 0:
            delta_vg_la = float(np.linalg.norm(virt_world - LA_world))
            print(f"step {k} | dist_to_goal={dist_to_goal:.3f} m")
            print(f"  sensors={sensor_vals}")
            print(f"  max_occ={max_occ:.3f}, cells>thr={num_thr}")
            print(
                f"  LA_world=({LA_world[0]:.3f},{LA_world[1]:.3f}), "
                f"VG_world=({virt_world[0]:.3f},{virt_world[1]:.3f}), "
                f"|VG-LA|={delta_vg_la:.3f}"
            )
            print(
                f"  F_att=({F_att[0]:.3f},{F_att[1]:.3f}), "
                f"F_rep=({F_rep[0]:.3f},{F_rep[1]:.3f})"
            )
            print(f"  uL={uL}, uR={uR}")
            print("")

        if dist_to_goal < 0.02:
            print(f"Reached goal within 2 cm at step {k}")
            break

        await client.sleep(dt)

    await node.set_variables(motors(0, 0))
    print("Stopped.")


# ------------------------------------------------------------------
# 6) Main entry point
# ------------------------------------------------------------------

async def main():
    client = ClientAsync()
    async with client:
        node = await client.wait_for_node()
        await node.lock()
        try:
            await follow_segment_with_avoid_debug(node, client)
        finally:
            await node.set_variables(motors(0, 0))
            await node.unlock()


if __name__ == "__main__":
    asyncio.run(main())
