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
        # defaults
        self.color_params = {"R_hmin": 0, "R_hmax": 10, "G_hmin": 50, "G_hmax": 70, "S_min": 100, "V_min": 100}
        self.areas_params = {"min_area": 500, "max_area": 10000}
        self.canny_params = {"sigma": 0.33, "low": 70, "high": 50}
        self.poly_params = {"min_area": 20}

    def set_params(self, params: dict):
        if "color_params" in params:
            self.color_params.update(params["color_params"])
        if "areas_params" in params:
            self.areas_params.update(params["areas_params"])
        if "canny_params" in params:
            self.canny_params.update(params["canny_params"])
        if "poly_params" in params:
            self.poly_params.update(params["poly_params"])

    def save(self):
        data = {
            "color_params": self.color_params,
            "areas_params": self.areas_params,
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
                self.areas_params.update(data.get("areas_params", {}))
                self.canny_params.update(data.get("canny_params", {}))
                self.poly_params.update(data.get("poly_params", {}))