import json
import os
from typing import Dict

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

        # Default empty params
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