import threading
import queue
import time
import cv2
import numpy as np
from shapely.geometry import Polygon

from computer_vision.vision_params_manager import VisionParamsManager
from computer_vision.computer_vision import ComputerVisionCore

class CameraCaptureThread(threading.Thread):
    """
    Background thread that continuously captures frames from a camera device
    and pushes the latest frame to a queue.Queue(maxsize=1). It never touches Tk.
    """
    def __init__(self, frame_queue: queue.Queue, device_index: int = 0, warmup_timeout: float = 2.0):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.device_index = device_index
        self._stop_event = threading.Event()
        self.warmup_timeout = warmup_timeout

        # Vision parameters manager
        self.params_manager = VisionParamsManager()

        # Store precomputed obstacles
        self.static_obstacles: list[Polygon] = []

    def stop(self):
        self._stop_event.set()

    def _grab_first_valid_frame(self, cap, timeout):
        """
        Try to grab a valid (non-black) frame within timeout seconds.
        Returns frame or None.
        """
        start = time.time()
        frame = None
        while time.time() - start < timeout and not self._stop_event.is_set():
            ok, frame = cap.read()
            if ok and frame is not None:
                # heuristics: not pure black
                if frame.sum() != 0:
                    return frame
            time.sleep(0.05)
        return None

    def run(self):
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            # push None to notify GUI if desired; we'll simply return
            cap.release()
            return

        # Warm-up: wait for the camera to produce a valid frame
        first_frame = self._grab_first_valid_frame(cap, self.warmup_timeout)
        if first_frame is not None:
            # Precompute static obstacles
            first_frame = ComputerVisionCore.detect_obstacles(
                first_frame,
                min_area=self.params_manager.poly_params["min_area"],
                canny_low=self.params_manager.canny_params["low"],
                canny_high=self.params_manager.canny_params["high"]
            )
            self.static_obstacles = first_frame["obstacles"]

            # Push first valid frame into queue
            try:
                self.frame_queue.put_nowait(first_frame)
            except queue.Full:
                try:
                    _ = self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(first_frame)
                except queue.Full:
                    pass

        # Continuous capture loop
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                # small sleep to avoid busy loop if camera fails temporarily
                time.sleep(0.05)
                continue

            # Process frame (e.g., detect robot)
            processed_frame = ComputerVisionCore.detect_robot(frame)

            # Overlay static obstacles on the frame
            overlay_frame = frame.copy()
            for poly in self.static_obstacles:
                pts = np.array(poly.exterior.coords, dtype=np.int32)
                cv2.polylines(overlay_frame, [pts], True, (0, 255, 0), 2)

            # Push latest frame into queue, keep only newest frame (maxsize=1)
            try:
                self.frame_queue.put_nowait(processed_frame)
            except queue.Full:
                try:
                    _ = self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(processed_frame)
                except queue.Full:
                    pass

            # throttle capture rate slightly to reduce CPU (adjust as needed)
            time.sleep(0.01)

        cap.release()