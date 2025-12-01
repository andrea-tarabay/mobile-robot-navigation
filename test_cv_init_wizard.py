import tkinter as tk
from src.computer_vision.vision_params_manager import VisionParamsManager
from src.computer_vision.computer_vision import ComputerVision
from src.gui.init_vision_wizard import InitVisionWizard
import cv2

# Load an image/frame for calibration
frame = cv2.imread("vision_live_output_pour_mehdi.jpg")

# Initialize params and CV object
params = VisionParamsManager()
params.load()  # load previous saved params if exist
cv_obj = ComputerVision(params)

# Launch your main Tkinter window
root = tk.Tk()
root.title("My Robot GUI")

# Button to open vision initialization wizard
def open_vision_wizard():
    InitVisionWizard(root, frame, params, cv_obj)

btn = tk.Button(root, text="Calibrate Vision", command=open_vision_wizard)
btn.pack()

root.mainloop()
