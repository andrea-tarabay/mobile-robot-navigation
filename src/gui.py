import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from fsm import Fsm

class Gui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Thymio on its treasure hunt")
        self.geometry("800x600")

        # --------------------------- Thread Management ---------------------------- #
        
        self.thread = Fsm(self.update_label)

        # --------------------------- UI Elements ---------------------------- #

        # --- Top frame with buttons ---
        top_frame = ttk.Frame(self)
        top_frame.pack(side="top", fill="x", padx=8, pady=8)

        self.start_btn = ttk.Button(top_frame, text="Start", command=self.on_start)
        self.start_btn.pack(side="left", padx=(0, 6))

        self.stop_btn = ttk.Button(top_frame, text="Stop", command=self.on_stop)
        self.stop_btn.pack(side="left")

        # Create a label to display the result
        self.label = ttk.Label(self, text="Result will appear here", font=("TkDefaultFont", 24))
        self.label.pack(padx=10 ,pady=10)

        # --- Frame reserved for the plot (fills rest of window) ---
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side="top", fill="both", expand=True, padx=8, pady=(0,8))

        # Create a matplotlib Figure and add a dummy sine plot
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._draw_dummy_plot()

        # Embed the matplotlib figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="top", fill="both", expand=True)

        # Optional toolbar (uncomment if you want panning/zoom buttons)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side="top", fill="x")


    def on_start(self):
        self.thread.start()
        print("Thread started.")

    def on_stop(self):
        self.thread.stop_work()
        print("Stop signal sent.")
        

    def update_label(self, number):
        # Must update UI from main thread
        self.after(0, lambda: self.label.config(text=f"Generated: {number}"))

    def _draw_dummy_plot(self):
        """Draw a dummy sine curve on the axes."""
        x = np.linspace(0, 2 * np.pi, 400)
        y = np.sin(x)

        self.ax.clear()
        self.ax.plot(x, y, label="sin(x)")

        # --- Make (0,0) top-left ----
        self.ax.set_xlim(0, max(x))
        self.ax.set_ylim(max(y), 0)  # invert y-axis -> origin at top-left

        self.ax.set_title("Dummy Sine Plot")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("sin(x)")

        self.ax.grid(True)
        self.ax.legend()
        
        # Important: draw the canvas if already embedded
        if hasattr(self, "canvas"):
            self.canvas.draw()