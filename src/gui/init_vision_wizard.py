import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

from src.computer_vision.vision_params_manager import VisionParamsManager
from src.computer_vision.computer_vision import ComputerVisionCore

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

        self.next_btn = ttk.Button(self, text="Next", command=self.next_step)
        self.next_btn.grid(row=1, column=3)

        self.prev_btn = ttk.Button(self, text="Back", command=self.prev_step)
        self.prev_btn.grid(row=1, column=2)

        self.update_step()

    # -------------------------------
    def next_step(self):
        if self.step < len(self.steps)-1:
            self.step+=1
            self.update_step()

    def prev_step(self):
        if self.step>0:
            self.step-=1
            self.update_step()
        
    # ------------------------
    def update_step(self):
        # destroy existing sliders
        for w in self.slider_vars.values():
            w["scale"].destroy()
        self.slider_vars.clear()

        step_name = self.steps[self.step]
        
        if step_name == "Detect Robot":
            pass  # no sliders for detect robot view
        elif step_name == "Canny":
            self.init_canny_sliders()
        elif step_name == "Polygons":
            self.init_polygon_sliders()
        #elif step_name == "Save":
        #    self.init_save_step()
        
        self.update_canvas()
        
    # -------------------------------
    def init_canny_sliders(self):
        row = 2
        var = tk.DoubleVar(value=self.params.canny_params.get("sigma"))
        scale = ttk.Scale(self, from_=0.1, to=1.0, orient="horizontal", variable=var, command=lambda e: self.update_canvas())
        scale.grid(row=row, column=0, columnspan=4, sticky="ew")
        tk.Label(self, text="sigma").grid(row=row, column=0)
        self.slider_vars["sigma"] = {"var": var, "scale": scale}
        row+=1

    # -------------------------------
    def init_polygon_sliders(self):
        row = 2
        for key, val in self.params.poly_params.items():
            var = tk.IntVar(value=val)
            scale = ttk.Scale(self, from_=1, to=400000, orient="horizontal", variable=var,
                              command=lambda e: self.update_canvas())
            scale.grid(row=row, column=0, columnspan=4, sticky="ew")
            tk.Label(self, text=key).grid(row=row, column=0, sticky="w")
            self.slider_vars[key] = {"var": var, "scale": scale}
            row += 1

    # -------------------------------
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

        # --- Update parameters from sliders ---
        for key, s in self.slider_vars.items():
            if key in self.params.canny_params:
                self.params.canny_params[key] = s["var"].get()
            if key in self.params.poly_params:
                self.params.poly_params[key] = int(s["var"].get())

        # --- Start from original frame ---
        frame_copy = self.frame.copy()

        # --- Process based on current step ---
        if self.steps[self.step] == "Detect Robot":
            # Automatically detect robot colors on raw frame
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
            polys = ComputerVisionCore.detect_polygons(frame_copy, 
                                                       self.params.poly_params.get("min_area"),
                                                       self.params.canny_params.get("low"),
                                                       self.params.canny_params.get("high"))

            for poly in polys:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(frame_copy, [pts], True, (0,255,0),2)

        # --- Resize & center ---
        rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                image=self.tk_img, anchor="center")
