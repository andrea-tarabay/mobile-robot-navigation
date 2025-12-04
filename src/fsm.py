import threading
import queue
import numpy as np
import time
import asyncio

from tdmclient import ClientAsync, aw

from pose_estimation.extended_kalman_filter import ExtendedKalmanFilter
from pose_estimation.differential_drive_system import DifferentialDriveSystem
from global_navigation.global_nav2 import GlobalNavigator
from computer_vision.computer_vision import ComputerVisionCore

# ------------------------------------------------------------
# Helper for path deviation measurement
# ------------------------------------------------------------

def distance_to_path(pose, path, max_pts=50):
    """Returns min dist from robot pose to polyline."""
    if path is None or len(path) < 2:
        return 9999

    x, y, _ = pose
    px = np.array([p[0] for p in path[:max_pts]])
    py = np.array([p[1] for p in path[:max_pts]])

    d = np.sqrt((px - x)**2 + (py - y)**2)
    return float(np.min(d))


class Fsm(threading.Thread):
    """
    Finite State Machine thread handling:
      - reading camera packets
      - EKF pose estimation
      - kidnapped robot detection
      - global path planning
      - local obstacle avoidance
      - motor command generation
    """

    def __init__(self, data_queue, 
                 #thymio_node, 
                 #client,
                 #ekf: ExtendedKalmanFilter,
                 #local_nav: "LocalAvoidance",
                 #g2g: "GoToGoalController",
                 dt=0.1,
                 ui_callback=None):
        super(Fsm, self).__init__(daemon=True)

        # FSM thread control flags
        self.__running = threading.Event() # ID used to stop the thread
        self.__resume = threading.Event() # ID used to pause the thread

        # modules for navigation
        #self.local_nav = local_nav
        #self.g2g = g2g

        # external interfaces
        self.data_queue = data_queue

        # internal state
        self.node = None
        self.client = None
        self.dt = dt
        self.current_path = None
        self.last_visible_pose = None
        self.kidnapped = False

        # EKF for pose estimation
        self.ekf = ExtendedKalmanFilter(
            mu0 = np.array([0.0, 0.0, 0.0]),
            Sigma0 = np.eye(3) * 1.0,
            system = DifferentialDriveSystem(dt=self.dt, 
                        lambda_=0.39735099337748336, 
                        axle_length=93.5, 
                        motion_noise_cov=np.eye(3)*0.03, 
                        measurement_noise_cov=np.eye(3)*0.01
                    )
        )

        # thresholds
        self.max_no_vision_time = 0.8  # seconds before declaring "kidnapped"
        self.last_vision_time = time.time()

        # Upstream interfaces callback
        self.ui_callback = ui_callback

    # ------------------------------------------------------------
    # Thread external interface (start/pause/stop)
    # ------------------------------------------------------------
    def start(self):
        self.__resume.set()
        self.__running.set()
        if not self.is_alive():
            super().start()

    def pause(self):
        self.__resume.clear()

    def resume(self):
        self.__resume.set()

    def stop(self):
        # stop loop
        self.__running.clear()
        self.__resume.set()

    def run(self):
        # new asyncio loop in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.async_run())

    # ------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------
    async def async_run(self):
        print("[FSM] Thread started.")

        self.client = ClientAsync()
        self.node = aw(self.client.wait_for_node())
        aw(self.node.lock())
        aw(self.node.wait_for_variables({"prox.horizontal", "motor.left.speed", "motor.right.speed"}))

        while self.__running.is_set():
            if not self.__resume.is_set():
                self.set_motors(left_target=0, right_target=0)
            self.__resume.wait()  # blocks when paused

            # --------------------------------------------------------
            # 1) Read vision packet
            # --------------------------------------------------------
            try:
                packet = self.data_queue.get(timeout=1.0)
            except queue.Empty:
                print("[FSM] Waiting for vision...")
                continue

            frame = packet["frame"]
            det = packet["detections"]

            robot_det = det["robot"]
            goal_det = det["goal"]
            obstacles = det["obstacles"]

            # --------------------------------------------------------
            # 2) Robotics sensors
            # --------------------------------------------------------
            sensor_vals = list(self.node.v.prox.horizontal)
            u_motor = np.array([
                self.node.v.motor.left.speed,
                self.node.v.motor.right.speed
            ])

            # --------------------------------------------------------
            # 3) EKF update → estimated pose
            # --------------------------------------------------------
            pose_pred, P_pred = self.ekf.predict(u_motor)

            if robot_det["found"]:
                self.last_vision_time = time.time()
                z = np.array([robot_det["center"][0],
                              robot_det["center"][1],
                              robot_det["theta"]])
                pose, P = self.ekf.update(z)
                self.last_visible_pose = pose
            else:
                # No new measurement
                pose, P = pose_pred, P_pred

            # --------------------------------------------------------
            # 4) Kidnapped robot detection
            # --------------------------------------------------------
            no_vision_elapsed = time.time() - self.last_vision_time

            if no_vision_elapsed > self.max_no_vision_time:
                if not self.kidnapped:
                    print("[FSM] Robot kidnapped! Using EKF only.")
                self.kidnapped = True
            else:
                if self.kidnapped:
                    print("[FSM] Vision restored! Checking if replanning is needed.")
                self.kidnapped = False

            # --------------------------------------------------------
            # 5) Decide if global path must be recomputed
            # --------------------------------------------------------
            need_replan = False

            if self.current_path is None:
                need_replan = True

            elif not self.kidnapped:
                # If robot deviates far from path after reappearing
                d = distance_to_path(pose, self.current_path)
                if d > 40:  # pixels threshold
                    print("[FSM] Robot off-path, replanning.")
                    need_replan = True

            if need_replan:
                if goal_det["found"]:
                    print("[FSM] Computing global path...")
                    result = GlobalNavigator.plan_path(
                        obstacles=obstacles,
                        start=pose[:2],
                        goal=goal_det["center"],
                        robot_radius_px=ComputerVisionCore.mm_to_px(ComputerVisionCore.ROBOT_RADIUS_MM),
                        img_width=frame.shape[1],
                        img_height=frame.shape[0],
                        debug_img=None
                    )

                    if result is not None:
                        self.current_path, _ = result
                    else:
                        print("[FSM] Path planning failed. Keeping current path.")
                else:
                    print("[FSM] Cannot plan: goal not visible.")

            # --------------------------------------------------------
            # 6) Local avoidance → virtual goal
            # --------------------------------------------------------
            """if self.current_path:
                virt_goal = self.local_nav.compute_virtual_goal(
                    pose, self.current_path, sensor_vals
                )
            else:
                virt_goal = None

            if virt_goal is None:
                print("[FSM] No virtual goal → stopping motors.")
                self.stop_motors()
                continue"""

            # --------------------------------------------------------
            # 7) Compute motor commands
            # --------------------------------------------------------
            """uL, uR, ctrl_info = self.g2g.compute_motor_commands(
                pose, virt_goal
            )"""

            # --------------------------------------------------------
            # 8) Apply to robot
            # --------------------------------------------------------
            self.set_motors(left_target=50, right_target=50)

            # --------------------------------------------------------
            # 9) Upstream interface callback
            # --------------------------------------------------------
            if self.ui_callback:
                self.ui_callback({
                    "pose": pose,
                    "pose_cov": P,
                    "kidnapped": self.kidnapped,
                    "path": self.current_path,
                    "goal": goal_det["center"] if goal_det["found"] else None,
                    "obstacle_count": len(obstacles),
                    "vision_found": robot_det["found"],
                    "timestamp": time.time(),
                })

            # Timing
            time.sleep(self.dt)

        # ------------------------------------------------------------
        # Cleanup on exit
        # ------------------------------------------------------------
        self.set_motors(left_target=0, right_target=0)
        aw(self.node.unlock())
        print("[FSM] Thread stopped.")

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------

    def set_motors(self, left_target=0, right_target=0):
        try:
            aw(self.node.set_variables({
                "motor.left.target":  [left_target],
                "motor.right.target": [right_target],
            }))
        except:
            pass