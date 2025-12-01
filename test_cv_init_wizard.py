import tkinter as tk
from src.computer_vision.vision_params_manager import VisionParamsManager
from src.computer_vision.computer_vision import ComputerVisionCore
from src.gui.init_vision_wizard import InitVisionWizard
import cv2

# Load an image/frame for calibration
frame = cv2.imread("vision_live_output_pour_jon.jpg")

# Initialize params and CV object
params = VisionParamsManager()

# Launch your main Tkinter window
root = tk.Tk()
root.title("My Robot GUI")

# Button to open vision initialization wizard
def open_vision_wizard():
    InitVisionWizard(root, frame, params)

btn = tk.Button(root, text="Calibrate Vision", command=open_vision_wizard)
btn.pack()

root.mainloop()
