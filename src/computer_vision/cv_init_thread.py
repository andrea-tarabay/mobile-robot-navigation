import threading
import cv2
import time

class CVInitThread(threading.Thread):
    """
    Thread for initializing the computer vision algorithm.
    
    After finishing, calls the provided callback on the main thread
    to notify the GUI that initialization is complete.
    """
    def __init__(self, callback):
        """
        Args:
            callback: function to call when initialization is done
        """
        super().__init__(daemon=True)
        self.callback = callback

    def run(self):
        try:
            # Open webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Cannot open camera.")
            
            # Read a frame
            time.sleep(0.1)  # Give some time for the camera to warm up
            ret, frame0 = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError("Impossible de lire la webcam.")

            # Initialize the vision algorithm
            # ... (initialization code here) ...

            # Call GUI callback safely
            self.callback(frame=frame0, error=None)

        except Exception as e:
            # Pass exception message to callback
            self.callback(frame=None, error=e)
