import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

from computer_vision.vision_params_manager import VisionParamsManager
from computer_vision.computer_vision import ComputerVisionCore

# =========================================================
# INIT VISION WIZARD (Tkinter)
# =========================================================
class InitVisionWizard(tk.Toplevel):
    def __init__(self, master, frame: np.ndarray, params: VisionParamsManager):
        super().__init__(master)

        self.frame = frame
        self.params = params
        self.step = 0
        self.steps = ["Detect Robot", "Canny", "Polygons", "Save"]

        # --- Window setup ---
        self.title("Vision Initialization Wizard")
        self.geometry("800x600")  # default window size
        self.minsize(600, 400)

        # Make rows/columns expandable
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # --- Canvas for displaying the image ---
        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=4, sticky="nsew")

        # Bind window resize to update image
        self.bind("<Configure>", lambda e: self.update_canvas())

        self.slider_vars = {}

        # --- Navigation buttons ---
        self.next_btn = ttk.Button(self, text="Next", command=self.next_step)
        self.next_btn.grid(row=1, column=3)

        self.prev_btn = ttk.Button(self, text="Back", command=self.prev_step)
        self.prev_btn.grid(row=1, column=2)

        self.save_btn = None

        self.update_step()

    # =========================================================
    # NAVIGATION
    # =========================================================
    def next_step(self):
        if self.step < len(self.steps) - 1:
            self.step += 1
            self.update_step()

    def prev_step(self):
        if self.step > 0:
            self.step -= 1
            self.update_step()
        
    # ------------------------
    def update_step(self):
        # Remove old sliders
        for w in self.slider_vars.values():
            w["scale"].destroy()
        self.slider_vars.clear()

        # Remove Save button if not needed
        if self.save_btn:
            self.save_btn.destroy()
            self.save_btn = None

        step_name = self.steps[self.step]
        
        # ----- Step UI -----
        if step_name == "Canny":
            self.init_canny_sliders()

        elif step_name == "Polygons":
            self.init_polygon_sliders()

        elif step_name == "Save":
            self.init_save_step()

        # ---- Button logic ----
        self.prev_btn.grid()
        if self.step == 0:
            self.prev_btn.grid_remove()

        if self.step == len(self.steps) - 1:
            self.next_btn.grid_remove()
        else:
            self.next_btn.grid()

        self.update_canvas()
        
    # =========================================================
    # CANNY SLIDERS
    # =========================================================
    def init_canny_sliders(self):
        row = 2
        var = tk.DoubleVar(value=self.params.canny_params.get("sigma"))

        scale = ttk.Scale(
            self, from_=0.1, to=1.0, variable=var,
            command=lambda e: self.update_canny_param("sigma")
        )
        scale.grid(row=row, column=0, columnspan=4, sticky="ew")

        tk.Label(self, text="sigma").grid(row=row, column=0, sticky="w")
        self.slider_vars["sigma"] = {"var": var, "scale": scale}

    def update_canny_param(self, key):
        """Fixes slider closure bug."""
        self.params.canny_params[key] = self.slider_vars[key]["var"].get()
        self.update_canvas()

    # =========================================================
    # POLYGON SLIDERS  (FIXED)
    # =========================================================
    def init_polygon_sliders(self):
        row = 2
        for key, val in self.params.poly_params.items():

            var = tk.IntVar(value=val)

            # IMPORTANT FIX: bind the key in the lambda
            scale = ttk.Scale(
                self,
                from_=1,
                to=4000,
                variable=var,
                command=lambda e, k=key: self.update_poly_param(k)
            )
            scale.grid(row=row, column=0, columnspan=4, sticky="ew")

            tk.Label(self, text=key).grid(row=row, column=0, sticky="w")

            self.slider_vars[key] = {"var": var, "scale": scale}
            row += 1

    def update_poly_param(self, key):
        """Fixes slider closure bug."""
        self.params.poly_params[key] = self.slider_vars[key]["var"].get()
        self.update_canvas()

    # =========================================================
    # FINAL SAVE STEP
    # =========================================================
    def init_save_step(self):
        label = tk.Label(self, text="Click save to store parameters.", font=("Arial", 16))
        label.grid(row=2, column=0, columnspan=4, pady=20)

        self.save_btn = ttk.Button(self, text="Save Parameters", command=self.save_params)
        self.save_btn.grid(row=3, column=0, columnspan=4)

    def save_params(self):
        self.params.save()
        self.destroy()

    # =========================================================
    # UPDATE CANVAS
    # =========================================================
    def update_canvas(self):
        """
        Update the canvas image based on the current wizard step.
        Handles dynamic resizing, image selection, overlays, and slider updates.
        """
        if self.frame is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width < 10 or canvas_height < 10:
            return

        # --- Start from original frame ---
        frame_copy = self.frame.copy()

        # --- Process based on current step ---
        if self.steps[self.step] == "Detect Robot":
            robot_state = ComputerVisionCore.detect_robot(frame_copy)
            if robot_state.get("found"):
                cv2.circle(frame_copy, robot_state["red_center"], 5, (0,0,255), -1)
                cv2.circle(frame_copy, robot_state["green_center"], 5, (0,255,0), -1)
                cv2.circle(frame_copy, robot_state["center"], 5, (255,0,0), -1)

        if self.steps[self.step] == "Canny":
            sigma = self.params.canny_params.get("sigma")
            low, high = ComputerVisionCore.init_canny(frame_copy, sigma)
            self.params.canny_params["low"] = low
            self.params.canny_params["high"] = high

            gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, low, high)
            frame_copy = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif self.steps[self.step] == "Polygons":
            obstacles = ComputerVisionCore.detect_obstacles(
                frame_copy, 
                self.params.poly_params.get("min_area"),
                self.params.canny_params.get("low"),
                self.params.canny_params.get("high")
            )

            for poly in obstacles:
                pts = np.array(poly.exterior.coords, dtype=np.int32)
                cv2.polylines(frame_copy, [pts], True, (0,255,0),2)

        # --- Resize and display ---
        rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                image=self.tk_img, anchor="center")
