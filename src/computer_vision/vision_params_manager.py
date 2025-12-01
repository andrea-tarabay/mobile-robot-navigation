import json
import os

# =========================================================
# PARAMETER MANAGEMENT
# =========================================================

class VisionParamsManager:
    def __init__(self, path="vision_params.json"):
        self.path = path
        # defaults
        self.color_params = {"R_hmin": 165, "R_hmax": 15, "G_hmin": 40, "G_hmax": 90}
        self.robot_area = {"min": 20, "max": 8000}
        self.canny_params = {"sigma": 0.33, "low": 70, "high": 50}
        self.poly_params = {"min_area": 20000, "max_area": 220000}

    def set_params(self, params: dict):
        if "color_params" in params:
            self.color_params.update(params["color_params"])
        if "robot_area" in params:
            self.robot_area.update(params["robot_area"])
        if "canny_params" in params:
            self.canny_params.update(params["canny_params"])
        if "poly_params" in params:
            self.poly_params.update(params["poly_params"])

    def save(self):
        data = {
            "color_params": self.color_params,
            "robot_area": self.robot_area,
            "canny_params": self.canny_params,
            "poly_params": self.poly_params
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
                self.color_params.update(data.get("color_params", {}))
                self.robot_area.update(data.get("robot_area", {}))
                self.canny_params.update(data.get("canny_params", {}))
                self.poly_params.update(data.get("poly_params", {}))