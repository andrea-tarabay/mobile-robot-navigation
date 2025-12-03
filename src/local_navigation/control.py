import math
import numpy as np
from dataclasses import dataclass

def wrap_to_pi(angle):
    """
    Wrap any angle (in radians) to the interval (-pi, pi].

    This avoids angles drifting to large values like 7π, -10π, etc.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

@dataclass
class GoToGoalGains:
    # Controller gains for the unicycle model
    Kv: float = 3.0      # Proportional gain for forward speed (distance -> v)
    Komega: float = 8.0  # Proportional gain for angular speed (heading error -> w)
    v_max: float = 0.12  # [m/s] maximum forward speed
    w_max: float = 4.0   # [rad/s] maximum angular speed

@dataclass
class ThymioKinematics:
    # Physical + conversion parameters of the Thymio robot
    wheel_radius: float = 0.022   # [m] radius of each wheel (~2.2 cm)
    half_axle: float   = 0.0475   # [m] half of the distance between the two wheels
    motor_gain: float  = 10.0     # factor to convert [rad/s] -> motor command units
    motor_max: int     = 300      # saturation limit on the motor command

class GoToGoalController:
    """
    Simple go-to-goal controller for a differential-drive robot.

    Input:
      - pose = [x, y, theta] in WORLD frame
      - goal = [xg, yg] in WORLD frame

    Output:
      - left, right motor commands (integers for Thymio)
    """

    def __init__(self,
                 gains: GoToGoalGains = GoToGoalGains(),
                 kin: ThymioKinematics = ThymioKinematics()):
        # Store gains + kinematic parameters
        self.g = gains
        self.kin = kin

    def compute_unicycle_cmd(self, pose, goal_xy):
        """
        Compute (v, omega) for the unicycle model that drives pose -> goal.

        pose    : [x, y, theta] in WORLD frame
        goal_xy : [xg, yg]      in WORLD frame

        Returns:
          v, w, (rho, alpha)

          where:
            rho   = distance to goal
            alpha = heading error (angle of goal in robot frame)
        """
        x, y, theta = pose
        xg, yg = goal_xy

        # Vector from robot to goal in WORLD frame
        dx = xg - x
        dy = yg - y

        # Distance to the goal
        rho = math.hypot(dx, dy)

        # Express that (dx, dy) vector in the ROBOT frame.
        # Here, convention is:
        #   y_forward = "forward" axis of the robot
        #   x_left    = "left" axis of the robot
        #
        # [y_forward]   [ cos(theta)   sin(theta)] [dx]
        # [x_left   ] = [-sin(theta)   cos(theta)] [dy]
        #
        # This is a rotation from WORLD -> ROBOT coordinates.
        y_forward =  math.cos(theta)*dx + math.sin(theta)*dy
        x_left    = -math.sin(theta)*dx + math.cos(theta)*dy

        # Heading error alpha: "angle to the goal in robot frame"
        # alpha > 0 => goal is to the left
        # alpha < 0 => goal is to the right
        alpha = math.atan2(x_left, y_forward)
        alpha = wrap_to_pi(alpha)

        # ---  dead-zone ---
        alpha_dead = 3.0 * math.pi / 180.0  # 3 degrees in radians
        if abs(alpha) < alpha_dead:
            alpha = 0.0
        # --------------------------

        # Proportional go-to-goal control:
        # - v increases with distance rho
        # - w increases with heading error alpha
        v = self.g.Kv * rho
        w = self.g.Komega * alpha

        # Saturate linear and angular speeds to avoid crazy values
        v = max(-self.g.v_max, min(self.g.v_max, v))
        w = max(-self.g.w_max, min(self.g.w_max, w))

        return v, w, (rho, alpha)

    def unicycle_to_wheels(self, v, w):
        """
        Map unicycle commands (v, w) to wheel commands (uL, uR)
        for the Thymio motors.

        v: linear velocity of robot center [m/s]
        w: angular velocity around vertical axis [rad/s]

        Returns integer motor commands uL, uR.
        """
        r = self.kin.wheel_radius
        L = self.kin.half_axle

        # Differential-drive kinematics:
        # v = (phi_R + phi_L)/2 * r
        # w = (phi_R - phi_L)/2 * r / L
        #
        # Solving for wheel angular velocities (phi_L, phi_R):
        phi_L  = (v / r) - (L / r) * w
        phi_R  = (v / r) + (L / r) * w

        # Convert [rad/s] into Thymio motor command units
        uL = int(self.kin.motor_gain * phi_L)
        uR = int(self.kin.motor_gain * phi_R)

        # Clip commands to motor limits
        uL = max(-self.kin.motor_max, min(self.kin.motor_max, uL))
        uR = max(-self.kin.motor_max, min(self.kin.motor_max, uR))

        return uL, uR

    def compute_motor_commands(self, pose, goal_xy, stop_if_close=True, tol=0.02):
        """
        High-level function to go directly from (pose, goal) -> (uL, uR)

        pose    : [x, y, theta] in WORLD frame
        goal_xy : [xg, yg]      in WORLD frame

        stop_if_close: if True, stop the robot when distance < tol
        tol          : distance threshold for "we've reached the goal" [m]

        Returns:
          (uL, uR), info

          where info is a dict:
             "rho"    : distance to goal
             "alpha"  : heading error
             "stopped": True if we decided to stop instead of move
        """
        # First compute ideal (v, w) for unicycle model
        v, w, (rho, alpha) = self.compute_unicycle_cmd(pose, goal_xy)

        # If we are very close to the goal, optionally stop completely
        if stop_if_close and rho < tol:
            return 0, 0, {"rho": rho, "alpha": alpha, "stopped": True}

        # Otherwise, convert (v, w) into wheel commands
        uL, uR = self.unicycle_to_wheels(v, w)
        return uL, uR, {"rho": rho, "alpha": alpha, "stopped": False}
