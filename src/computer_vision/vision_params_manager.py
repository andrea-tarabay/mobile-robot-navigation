import json
import os

# =========================================================
# CONSTANTS
# =========================================================
DEFAULT_FILE = "vision_params.json"

# =========================================================
# PARAMETER MANAGEMENT
# =========================================================

class VisionParamsManager:
    def __init__(self, path=DEFAULT_FILE):
        self.path = path

        self.color_params = {}
        self.areas_params = {}
        self.canny_params = {}
        self.poly_params = {}
        self.robot_params = {}


    def set_params(self, params: dict):
        if "color_params" in params:
            self.color_params.update(params["color_params"])
        if "areas_params" in params:
            self.areas_params.update(params["areas_params"])
        if "canny_params" in params:
            self.canny_params.update(params["canny_params"])
        if "poly_params" in params:
            self.poly_params.update(params["poly_params"])
        if "robot_params" in params:
            self.robot_params.update(params["robot_params"])

    def save(self):
        data = {
            "color_params": self.color_params,
            "areas_params": self.areas_params,
            "canny_params": self.canny_params,
            "poly_params": self.poly_params,
            "robot_params": self.robot_params
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
                self.set_params(data)

    def set_defaults(self):
        self.color_params = {
            "R_hmin": 165,
            "R_hmax": 15,
            "G_hmin": 40,
            "G_hmax": 90,
            "S_min": 40,
            "V_min": 40
        }
        self.areas_params = {
            "min_obstacle_area": 20000,
            "min_red_area": 150,
            "max_red_area": 2000,
            "min_green_area": 150,
            "max_green_area": 2000,
            "min_blue_area": 150,
            "max_blue_area": 2000,
            "min_robot_area": 5000,
            "max_robot_area": 50000,
            "margin": 10
        }
        self.canny_params = {
            "sigma": 0.33,
            "low": 100,
            "high": 200
        }
        self.poly_params = {
            "min_area": 3943
        }
        self.robot_params = {
            "robot_length_mm": 100,
            "robot_width_mm": 90,
            "robot_radius_mm": 110
        }