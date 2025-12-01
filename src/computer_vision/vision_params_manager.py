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
        self.canny_params = {"sigma": 0.33, "low": 70, "high": 50}
        self.poly_params = {"min_area": 20}

    def set_params(self, params: dict):
        if "canny_params" in params:
            self.canny_params.update(params["canny_params"])
        if "poly_params" in params:
            self.poly_params.update(params["poly_params"])

    def save(self):
        data = {
            "canny_params": self.canny_params,
            "poly_params": self.poly_params
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
                self.canny_params.update(data.get("canny_params", {}))
                self.poly_params.update(data.get("poly_params", {}))