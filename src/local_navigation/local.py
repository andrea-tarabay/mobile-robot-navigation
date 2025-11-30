import math
from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from .local_occupancy import (
    sensor_measurements,
    sensor_distances,
    thymio_coords,
    sensor_pos_from_center,
    sensor_angles,
)



def variable_info(variable):
    """
    Provided a variable, prints the type and content of the variable
    """
    print("This variable is a {}".format(type(variable)))
    if type(variable) == np.ndarray:
        print("\n\nThe shape is {}".format(variable.shape))
    print("\n\nThe data contained in the variable is : ")
    print(variable)
    print("\n\nThe elements that can be accessed in the variable are :\n")
    print(dir(variable))
    
variable_info(np.array([1]))


## Interpolation from sensor values to distances in cm
def sensor_val_to_cm_dist(val):
    """
    Returns the distance corresponding to the sensor value based 
    on the sensor characteristics
    :param val: the sensor value that you want to convert to a distance
    :return: corresponding distance in cm
    """
# Some sensors report 0 when nothing is detected (out of range).
    # Treat that as "infinite" distance instead of trying to interpolate.
    if val == 0:
        return np.inf

    # Build a 1D interpolation function from calibration data.
    # interp1d will linearly interpolate y for any x inside the data range.
    # NOTE: By default, it raises a ValueError for x outside the range.
    # If you want extrapolation, use: interp1d(..., fill_value="extrapolate").
    f = interp1d(sensor_measurements, sensor_distances)

    # Evaluate the interpolation at the requested sensor value.
    # .item() returns a plain Python float from the 0-D NumPy result.
    return f(val).item()



# Raw Thymio prox -> obstacle positions around robot (cm)
# ------------------------------------------------------------

def obstacles_pos_from_sensor_vals(sensor_vals):
    """
    Convert Thymio's horizontal proximity values to obstacle positions
    in the ROBOT frame, in centimeters.

    Robot frame (matching course plots):
      - x = sideways (left +, right -)
      - y = forward  (in front of Thymio +)

    Assumes:
      - sensor_val_to_cm_dist(val)
      - sensor_angles          (angle of each sensor wrt forward)
      - sensor_pos_from_center (position of each sensor on Thymio)
    """
    # 1) raw value -> distance in cm (np.inf if nothing detected)
    dists_cm = [sensor_val_to_cm_dist(v) for v in sensor_vals]

    # 2) project along sensor angle (robot frame)
    dx = [d * math.cos(a) for d, a in zip(dists_cm, sensor_angles)]
    dy = [d * math.sin(a) for d, a in zip(dists_cm, sensor_angles)]

    # 3) shift from sensor head to robot center
    obstacles = []
    for (sx, sy), ddx, ddy in zip(sensor_pos_from_center, dx, dy):
        # sx, sy in cm; ddx, ddy in cm
        obstacles.append([sx + ddx, sy + ddy])

    return np.array(obstacles, dtype=float)  # shape (Nsensors, 2), in cm


# ------------------------------------------------------------
# local occupancy grid around Thymio (in meters)
# left + right - front + back -
# ------------------------------------------------------------

class LocalOccupancyGrid:
    """
    Robot-centred occupancy grid.

    - x axis: sideways (left +, right -)
    - y axis: forward (front +, back -)
    - Units inside: meters
    """

    def __init__(self,
                 size_m=0.15,        # grid side [m] 0.6 x 0.6 m (60cm x 60cm)
                 resolution_m=0.001, # cell size [m] 2cmx2cm
                 hit_inc=0.1,       # increment per hit
                 decay=0.98):       # to forget the obstacles over time 
        self.size_m     = size_m
        self.resolution = resolution_m
        self.hit_inc    = hit_inc
        self.decay      = decay

        self.half_size  = size_m / 2.0  #how far the gird extends from the center   
        self.n_cells    = int(size_m / resolution_m) #30 x30 matrix cells 
        self.grid       = np.zeros((self.n_cells, self.n_cells), dtype=float) #Occupancy grid

    # ---- coord <-> cell index ----

    def _point_to_index(self, x_m, y_m): 
        """
        Robot-frame point [m] -> (iy, ix) index.
        x_m: sideways, y_m: forward.
        """
        if abs(x_m) > self.half_size or abs(y_m) > self.half_size:
            return None # outside grid

        ix = int((x_m + self.half_size) / self.resolution)  #column index
        iy = int((y_m + self.half_size) / self.resolution) #row index

        ix = min(max(ix, 0), self.n_cells - 1) #safety clamp in case of rounding errors 
        iy = min(max(iy, 0), self.n_cells - 1)
        return iy, ix  #row, column

    def _index_to_point(self, iy, ix): #getting point from robot frame from cell index 
        """
        Cell index -> robot-frame point [m] at cell center.
        """
        x_m = (ix + 0.5) * self.resolution - self.half_size
        y_m = (iy + 0.5) * self.resolution - self.half_size
        return x_m, y_m

    # ---- main update from prox sensors ----

    def update_from_sensor_vals(self, sensor_vals):
        """
        Update grid from Thymio horizontal proximity sensor readings.
        """
        # Forget old stuff a bit
        self.grid *= self.decay 

        # obstacles around robot in cm
        obs_cm = obstacles_pos_from_sensor_vals(sensor_vals)

        for ox_cm, oy_cm in obs_cm:
            if not np.isfinite(ox_cm) or not np.isfinite(oy_cm):
                continue

            # cm -> m
            x_m = ox_cm / 100.0
            y_m = oy_cm / 100.0

            idx = self._point_to_index(x_m, y_m)
            if idx is None:
                continue

            iy, ix = idx
            self.grid[iy, ix] = min(1.0, self.grid[iy, ix] + self.hit_inc)



# ------------------------------------------------------------
#  local navigation config (tunable parameters)
# ------------------------------------------------------------

class LocalNavConfig:
    def __init__(self,
                 lookahead_dist_m=0.15,
                 max_lookahead_points=15,
                 occ_threshold=0.5,
                 rep_influence_radius=0.13,
                 k_att=1.0,
                 k_rep=0.6,
                 virt_goal_dist_m=0.10):
        """
        lookahead_dist_m      : how far along path to look ahead
        max_lookahead_points  : max indices to move ahead in path array
        occ_threshold         : cell value above this considered 'occupied'
        rep_influence_radius  : [m] radius where obstacles repel
        k_att                 : gain on attraction to path
        k_rep                 : gain on obstacle repulsion
        virt_goal_dist_m      : [m] distance of virtual goal in front of robot
        """
        self.lookahead_dist_m     = lookahead_dist_m
        self.max_lookahead_points = max_lookahead_points
        self.occ_threshold        = occ_threshold
        self.rep_influence_radius = rep_influence_radius
        self.k_att                = k_att
        self.k_rep                = k_rep
        self.virt_goal_dist_m     = virt_goal_dist_m


# ------------------------------------------------------------
# local navigation on top of grid + global path
# ------------------------------------------------------------

class LocalNavigator:
    """
    Takes:
      - pose [x, y, theta] in WORLD frame
      - global path (N x 2) in WORLD frame
      - LocalOccupancyGrid (around robot)

    Returns:
      - virtual goal [xg, yg] in WORLD frame
    """

    def __init__(self, grid: LocalOccupancyGrid, cfg: LocalNavConfig):
        self.grid     = grid
        self.cfg      = cfg
        self.path_idx = 0  # progress along global path

    # ---------- 4.1 World <-> Robot transforms ----------

    @staticmethod
    def world_to_robot(pose, point_world):
        """
        WORLD -> ROBOT frame.

        pose        = [x_r, y_r, theta_r] robot pose in thr world frame 
        point_world = [x_w, y_w] point in world frame 

        Robot:
          x_robot = left  (+)
          y_robot = fwd   (+)
        """
        x_r, y_r, theta = pose
        px, py = point_world

        dx = px - x_r #moving origin to robot 
        dy = py - y_r

        c = math.cos(theta)
        s = math.sin(theta)

        y_forward =  c*dx + s*dy #reorienting axes into robot axes
        x_left    = -s*dx + c*dy

        return np.array([x_left, y_forward], dtype=float)

    @staticmethod
    def robot_to_world(pose, point_robot):
        """
        ROBOT -> WORLD frame.

        pose        = [x_r, y_r, theta_r]
        point_robot = [x_left, y_forward]
        """
        x_r, y_r, theta = pose
        x_left, y_forward = point_robot

        c = math.cos(theta)
        s = math.sin(theta)

        dx_w = -s*x_left + c*y_forward
        dy_w =  c*x_left + s*y_forward

        return np.array([x_r + dx_w, y_r + dy_w], dtype=float)

    # ---------- 4.2 Find look-ahead point on path ----------

    def _find_lookahead_point(self, pose, path):
        """
        Returns (lookahead_point_world, closest_idx)
        """
        x, y, theta = pose # robot pose in world frame
        if path is None or path.shape[0] == 0:
            return None, None 

        start_idx = self.path_idx # continue from last progress
        end_idx   = min(path.shape[0], start_idx + self.cfg.max_lookahead_points) # furthest lookahead index
        segment   = path[start_idx:end_idx] #points with indices 
        dists2    = (segment[:, 0] - x)**2 + (segment[:, 1] - y)**2 # squared distances to robot
        k_local   = int(np.argmin(dists2)) #find closest point in segment
        k_global  = start_idx + k_local # locating robot approximately on full path
        
        # Update progress to guarantee we dont go backwards
        self.path_idx = k_global

        # Move forward along path until lookahead_dist is reached
        target_idx = k_global
        accum = 0.0 #how much distance we have accumulated along the path
        while (target_idx + 1 < path.shape[0] and # making sure we dont go out of bounds
               accum < self.cfg.lookahead_dist_m and # distance not yet reached
            target_idx - k_global < self.cfg.max_lookahead_points): # max lookahead points
            p0 = path[target_idx] #current point
            p1 = path[target_idx + 1] #next point on the path 
            step = float(np.linalg.norm(p1 - p0)) # distance between current and next point
            accum += step # accumulate distance
            target_idx += 1 # move to next point
        #target_idx is the index of selected lookahead point 
        LA_point = path[target_idx] # lookahead point in world frame
        return LA_point, k_global 

    # ---------- 4.3 Corridor check to LA ----------

    def _corridor_free_to_LA(self, p_LA_robot):
        """
        Straight line from (0,0) to p_LA_robot in ROBOT frame.
        Return True if no occupied cells on that line.
        """
        # If LA is behind or exactly at us, consider blocked
        if p_LA_robot[1] <= 0.0:
            return False

        n_samples = 10
        for i in range(1, n_samples + 1):
            alpha = i / n_samples
            pt = alpha * p_LA_robot  # [x_left, y_forward] in robot frame
            x_m, y_m = float(pt[0]), float(pt[1])

            idx = self.grid._point_to_index(x_m, y_m)
            if idx is None:
                continue

            iy, ix = idx
            if self.grid.grid[iy, ix] > self.cfg.occ_threshold:
                return False

        return True

    # ---------- 4.4 Repulsive vector from occupancy ----------

    def _repulsive_vector(self):
        """
        Compute repulsive force in ROBOT frame [x_left, y_forward].
        """
        F_rep = np.zeros(2, dtype=float)
        R = self.cfg.rep_influence_radius
        eps = 1e-3

        occ_idxs = np.argwhere(self.grid.grid > self.cfg.occ_threshold)

        for iy, ix in occ_idxs:
            x_m, y_m = self.grid._index_to_point(iy, ix)
            d = math.sqrt(x_m*x_m + y_m*y_m)

            if d < eps or d > R:
                continue

            # Direction from obstacle -> robot
            dir_vec = np.array([-x_m / d, -y_m / d])
            # Magnitude stronger when closer, zero at R
            mag = self.cfg.k_rep * (1.0 / d - 1.0 / R)

            F_rep += mag * dir_vec

        return F_rep

    # ---------- 4.5 MAIN: compute virtual goal ----------

    def compute_virtual_goal(self, pose, path):
        """
        pose : [x, y, theta] in WORLD frame
        path : (N, 2) array in WORLD frame

        Returns: [xg, yg] in WORLD frame or None.
        """
        if path is None or path.shape[0] == 0:
            return None

        # 1) Choose look-ahead on path
        p_LA_world, _ = self._find_lookahead_point(pose, path)
        if p_LA_world is None:
            return None

        # 2) Express LA in ROBOT frame
        p_LA_robot = self.world_to_robot(pose, p_LA_world)

        # 3) If straight path to LA is free, use LA directly
        if self._corridor_free_to_LA(p_LA_robot):
            return p_LA_world.copy()

        # 4) Else: compute potential-field direction

        # Attractive towards LA
        F_att = p_LA_robot.astype(float)
        norm_att = np.linalg.norm(F_att)
        if norm_att > 1e-6:
            F_att = (self.cfg.k_att / norm_att) * F_att

        # Repulsive from obstacles
        F_rep = self._repulsive_vector()

        F_tot = F_att + F_rep
        norm_tot = np.linalg.norm(F_tot)
        if norm_tot < 1e-6:
            # fallback: just forward
            F_tot = np.array([0.0, 1.0], dtype=float)
            norm_tot = 1.0

        dir_robot = F_tot / norm_tot  # unit vector in robot frame

        # Place virtual goal some distance ahead in that direction
        virt_robot = self.cfg.virt_goal_dist_m * dir_robot

        # 5) Convert back to WORLD frame
        virt_world = self.robot_to_world(pose, virt_robot)
        return virt_world

