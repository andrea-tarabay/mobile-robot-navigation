# Local Navigation & Control Module

This module implements the **local navigation stack** for a Thymio-like robot:

1. Read proximity sensors.
2. Convert sensor values into obstacle positions in the **robot frame**.
3. Update a robot-centred **occupancy grid** with inflation (robot radius).
4. Given a global **path** (sequence of waypoints), compute a **lookahead point**.
5. Combine attraction to the path and repulsion from obstacles to get a **virtual goal**.
6. Use a unicycle **go-to-goal controller** to turn that virtual goal into wheel motor commands.

---

## Coordinate Frames

- **World frame**: `(x, y)` in meters, map coordinates.
- **Robot frame**:
  - `x_left  > 0`  → obstacle on the **left** of the robot  
  - `x_left  < 0`  → obstacle on the **right**  
  - `y_forward > 0` → obstacle **in front** of the robot  

All local navigation and occupancy grid computations happen in the robot frame.

---

## Utility Functions

### `wrap_to_pi(angle: float) -> float`
Wrap a radian angle into the interval `(-π, π]`.  
Used to keep heading errors bounded.

### `sensor_val_to_cm_dist(val: int) -> float`
Interpolate a proximity sensor value into a **distance in cm** using the calibration
arrays `sensor_measurements` and `sensor_distances`.  
Returns `np.inf` when `val == 0` (no detection).

### `obstacles_pos_from_sensor_vals(sensor_vals)`
Convert horizontal proximity readings into obstacle positions in the **robot frame**:

1. Convert each raw sensor value to a distance (cm).
2. For each sensor, compute the obstacle position in the original frame
   `(x_forward, y_left)` using the sensor’s pose and angle.
3. Re-express it in the robot frame `(x_left, y_forward)`.
4. Return a `numpy` array of obstacle positions `[x_left, y_forward]` in cm.

---

## `LocalOccupancyGrid`

Robot-centred occupancy grid around the robot.

### `__init__(size_m, resolution_m, hit_inc, decay, robot_radius_m)`
Create a square grid of side `size_m` with cell size `resolution_m`:

- `hit_inc`: how much to increase occupancy per obstacle hit.
- `decay`: multiplicative factor applied each update to fade old obstacles.
- `robot_radius_m`: used to inflate occupied cells (safety margin).

### `_point_to_index(x_m, y_m)`
Convert a point `(x, y)` in meters (robot frame) to grid indices `(iy, ix)`.
Returns `None` if the point is outside the grid.

### `_index_to_point(iy, ix)`
Inverse of `_point_to_index`: returns the cell center `(x_m, y_m)` in meters.

### `update_from_sensor_vals(sensor_vals)`
Main grid update:

1. Apply decay: `grid *= decay`.
2. Compute obstacle positions (cm) from sensor values.
3. Convert to meters and project into grid indices.
4. For each obstacle cell, **inflate** occupancy in a disk of radius `robot_radius_m`
   (in grid cells) so the robot keeps a safety margin.
5. Clamp occupancy values to `≤ 1.0`.

---

## `LocalNavConfig` (dataclass)

Configuration parameters for local navigation:

- `lookahead_dist_m`: arc length along the path to look ahead from the closest waypoint.
- `max_lookahead_points`: maximum number of path points to search/advance.
- `occ_threshold`: minimum occupancy for a cell to generate repulsion.
- `rep_influence_radius`: radius (m) around the robot where obstacles produce repulsive force.
- `k_att`: gain (norm) of the attractive force toward the lookahead.
- `k_rep`: gain of the repulsive force from obstacles.
- `virt_goal_dist_m`: distance (m) from robot to the virtual goal in the final direction.

---

## `LocalNavigator`

Performs path following with obstacle avoidance.

### `__init__(grid: LocalOccupancyGrid, cfg: LocalNavConfig)`
Stores references to the occupancy grid and config, and initializes `path_idx`
(the index of the current position along the path).

### `world_to_robot(pose, point_world)`
Convert a point from world frame to robot frame `(x_left, y_forward)` given
`pose = (x_r, y_r, θ)`.

### `robot_to_world(pose, point_robot)`
Inverse transform: convert `(x_left, y_forward)` back to world coordinates `(x, y)`.

### `_find_lookahead_point(pose, path)`
Given the robot pose and a `numpy` path `path` (N×2):

1. Clamp `self.path_idx` to `[0, N-1]`.
2. Take a path segment starting at `path_idx` of length at most `max_lookahead_points`.
3. Find the closest point in that segment to the current pose.
4. Advance along the path from this closest point, accumulating segment lengths
   until `lookahead_dist_m` or the end of the path is reached.
5. Return:
   - `LA_point`: chosen lookahead waypoint.
   - `k_global`: index of the closest waypoint.

If the segment is empty, returns the last path point.

### `_repulsive_vector(self)`
Compute the **repulsive force** in robot frame:

1. Iterate over all grid cells with occupancy > `occ_threshold`.
2. Convert each cell index back to a point `(x_m, y_m)` in the robot frame.
3. Ignore cells behind the robot (`y_m <= 0`) or farther than `rep_influence_radius`.
4. Compute a unit vector pointing **away** from the obstacle.
5. Weight by occupancy and a distance-based factor `s²` (stronger when closer).
6. Sum all contributions into `F_rep`.
7. Saturate the final norm so repulsion cannot dominate attraction by more than
   `max_F_rep ≈ 1.85 * k_att`.

Returns `F_rep = [Fx_left, Fy_forward]`.

### `compute_virtual_goal(self, pose, path, sensor_vals=None)`
Compute a **virtual goal** to feed to the go-to-goal controller:

1. Early exit if `path` is empty.
2. Convert `path` to a `numpy` array and find `LA_world` via `_find_lookahead_point`.
3. Transform lookahead into robot frame: `p_LA_robot`.
4. Build attractive force `F_att` of unit norm scaled by `k_att`.
5. Compute repulsive force `F_rep` from the occupancy grid.
6. If `sensor_vals` are absent or all readings are low (`max(sensor_vals) < 300`),
   there are no important obstacles → **go straight to lookahead**:
   returns `(LA_world, LA_world, F_att, 0)`.
7. Otherwise:
   - Sum forces: `F_tot = F_att + F_rep`.
   - Normalize to a unit direction.
   - Create a small step of length `virt_goal_dist_m` in that direction.
   - Convert back to world frame as `virt_world`.
   - Return `(virt_world, LA_world, F_att, F_rep)`.

---

## Go-to-Goal Controller

### `GoToGoalGains` (dataclass)
- `Kv`: gain on distance to goal.
- `Komega`: gain on heading error.
- `v_max`, `w_max`: saturation limits for linear and angular velocities.

### `ThymioKinematics` (dataclass)
Physical / actuation parameters:

- `wheel_radius`: radius of each wheel (m).
- `half_axle`: half the distance between wheels (m).
- `motor_gain`: conversion gain from wheel angular velocity to motor command.
- `motor_max`: saturation on motor command.

### `GoToGoalController`

High-level goal: given `pose` and a **goal point in world frame**, output
left/right motor commands.

#### `compute_unicycle_cmd(self, pose, goal_xy)`
1. Compute position error `(dx, dy)` in world coordinates.
2. Express error in robot frame to get:
   - `rho`: distance to goal.
   - `alpha`: heading error (angle between robot’s forward axis and the goal).
3. Apply a dead zone on `alpha`.
4. Compute linear and angular velocities `v`, `w` using `Kv` and `Komega`.
5. Saturate `v` and `w` to `v_max`, `w_max`.
6. Return `(v, w, rho, alpha)`.

#### `unicycle_to_wheels(self, v, w)`
Convert unicycle `(v, w)` to wheel angular velocities:

- `phi_L = (v / r) - (L / r) * w`
- `phi_R = (v / r) + (L / r) * w`

Then scale to motor commands with `motor_gain` and clamp to `[-motor_max, motor_max]`.  
Returns `(uL, uR)`.

#### `compute_motor_commands(self, pose, goal_xy, tol=0.02, stop_if_close=True)`
Wrapper that:

1. Calls `compute_unicycle_cmd`.
2. If `rho < tol` and `stop_if_close`, returns zero commands and a small info dict.
3. Otherwise converts to wheel commands and returns `(uL, uR, info)`.

---

## Default Instances

At the end of the file, the following defaults are instantiated:

- `grid = LocalOccupancyGrid(...)`: occupancy grid around the robot.
- `cfg = LocalNavConfig(...)`: navigation parameters.
- `navigator = LocalNavigator(grid, cfg)`: local navigation engine.
- `gains = GoToGoalGains(...)`: controller gains.
- `kin = ThymioKinematics()`: robot kinematics.
- `g2g = GoToGoalController(gains, kin)`: go-to-goal controller used by the FSM.

These objects are meant to be imported and used directly by the higher-level
finite state machine (FSM) that drives the robot.