import platform
import cv2
import os

def find_available_camera(max_test=5):
    """
    Robust camera detection that works for:
    - macOS (always index 0 first)
    - Windows internal+external webcams (prefer external index 1)
    - Linux (generic scanning)
    
    Priority:
        1) CAM_INDEX env variable (manual override)
        2) macOS -> try index 0
        3) Windows -> try index 1 then 0
        4) Scan all indices [0..max_test-1]
    """

    # ------------------------------
    # 1) Manual override
    # ------------------------------
    if "CAM_INDEX" in os.environ:
        try:
            env_idx = int(os.environ["CAM_INDEX"])
            cap = cv2.VideoCapture(env_idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return env_idx, cap
            cap.release()
        except:
            pass

    system = platform.system()

    # ------------------------------
    # 2) macOS: always start with 0
    # ------------------------------
    if system == "Darwin":
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return 0, cap
        cap.release()

    # ------------------------------
    # 3) Windows: external USB often index 1
    # ------------------------------
    elif system == "Windows":
        # Try USB webcam first
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return 1, cap
        cap.release()

        # Try internal laptop webcam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return 0, cap
        cap.release()

    # ------------------------------
    # 4) Fallback: scan all camera indices
    # ------------------------------
    for idx in range(max_test):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return idx, cap
        cap.release()

    return None, None