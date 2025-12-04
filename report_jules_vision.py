"""
Concise vision pipeline report + notebook-ready helpers.

- Prints a short markdown summary of the pipeline and key functions.
- Runs the CV pipeline on a frame and returns intermediate images.
- Offers quick plotting helpers for Jupyter (shows each step side by side).
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Ensure local src/ is on the import path when used from a notebook.
REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from computer_vision.vision import Vision
from computer_vision.computer_vision import ComputerVisionCore
from computer_vision.vision_params_manager import VisionParamsManager
from utils.camera_utils import find_available_camera

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

PIPELINE_TEXT = dedent(
    """
    ## Vision pipeline (résumé court)
    1) Acquisition : `find_available_camera` cherche l’indice valable (env -> mac -> win -> scan).  
    2) Robot : HSV sur rouge/vert (`detect_robot`), sélection des cercles, paire scorée, centre + theta.  
    3) Goal : HSV sur bleu (`detect_goal`), cercle unique.  
    4) Bords/obstacles : Canny auto (`init_canny`), contours filtrés par aire, convex hull -> polygones; on ignore la zone robot/goal.  
    5) Wrapper : `Vision` applique lissage optionnel + cache.  
    6) Calibrage : `InitVisionWizard` (Tk) règle HSV/Canny/polygones et sauvegarde `vision_params.json`.

    ## Fonctions clés (où regarder)
    - `utils.camera_utils.find_available_camera` : ordre de priorité des webcams + override `CAM_INDEX`.
    - `computer_vision.computer_vision.detect_robot` : masques HSV rouge/vert, candidats cercles, scoring de paire, retourne centre/theta + `robot_mask`.
    - `computer_vision.computer_vision.detect_goal` : masque HSV bleu, centre du but + `goal_mask`.
    - `computer_vision.computer_vision.detect_obstacles` : Canny (seuils auto), filtres d’aire, convex hull, exclut boîtes robot/goal.
    - `computer_vision.computer_vision.init_canny` : seuils bas/haut dérivés de la médiane de l’image.
    - `computer_vision.vision.Vision` : cache + lissage exponentiel pour robot/goal, conversions px/mm utilitaires.
    - `gui.init_vision_wizard.InitVisionWizard` : sliders HSV/Canny/polygones, aperçu live par étape, sauvegarde des params.
    """
).strip()


def make_report_markdown() -> str:
    """Return the markdown text (for display in Jupyter)."""
    return PIPELINE_TEXT


def display_report():
    """
    Display the markdown report in Jupyter; fall back to plain print elsewhere.
    """
    try:
        from IPython.display import Markdown, display  # type: ignore

        display(Markdown(make_report_markdown()))
    except Exception:
        print(make_report_markdown())


# ---------------------------------------------------------------------------
# Helpers to run the pipeline and collect intermediate images
# ---------------------------------------------------------------------------

def load_params(params_path: str = "vision_params.json") -> VisionParamsManager:
    mgr = VisionParamsManager(params_path)
    mgr.load()
    return mgr


def grab_single_frame(device_index: Optional[int] = None) -> np.ndarray:
    """
    Capture one frame from camera. If device_index is None, auto-detect.
    """
    if device_index is None:
        idx, cap = find_available_camera()
        if cap is None:
            raise RuntimeError("No camera found by find_available_camera.")
    else:
        idx, cap = device_index, cv2.VideoCapture(device_index)

    if cap is None or not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {idx}.")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Failed to read a frame from camera.")
    return frame


def run_pipeline(
    frame_bgr: np.ndarray,
    params: VisionParamsManager,
    smoothing_alpha: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Run CV on a BGR frame and return intermediate visualizations.
    Keys: original, robot_mask, goal_mask, edges, obstacles_overlay.
    """
    vision = Vision(
        params_color=params.color_params,
        params_canny=params.canny_params,
        params_poly=params.poly_params,
        smoothing_alpha=smoothing_alpha,
    )

    outputs: Dict[str, np.ndarray] = {"original": frame_bgr.copy()}

    # Robot and goal
    robot_state = vision.detect_robot(frame_bgr.copy(), use_cache=False)
    goal_state = vision.detect_goal(frame_bgr.copy(), use_cache=False)
    outputs["robot_mask"] = robot_state.get("robot_mask")
    outputs["goal_mask"] = goal_state.get("goal_mask")

    # Canny edges
    sigma = params.canny_params.get("sigma", 0.33)
    low, high = ComputerVisionCore.init_canny(frame_bgr, sigma)
    edges = cv2.Canny(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), low, high)
    outputs["edges"] = edges

    # Obstacles overlay
    obstacle_overlay = frame_bgr.copy()
    obstacles = vision.detect_obstacles(frame_bgr.copy(), use_cache=False)
    for poly in obstacles:
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.polylines(obstacle_overlay, [pts], True, (0, 255, 0), 2)
    if robot_state.get("found"):
        cv2.circle(obstacle_overlay, robot_state["center"], 6, (0, 0, 255), -1)
    if goal_state.get("found"):
        cv2.circle(obstacle_overlay, goal_state["center"], 6, (255, 255, 0), -1)
    outputs["obstacles_overlay"] = obstacle_overlay

    return outputs


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR/gray to RGB for matplotlib display."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_steps(steps: Dict[str, np.ndarray], title: str = "Vision pipeline"):
    """
    Plot available steps side by side (Jupyter-friendly).
    """
    ordered_keys = ["original", "robot_mask", "goal_mask", "edges", "obstacles_overlay"]
    panels: list[Tuple[str, np.ndarray]] = []
    for k in ordered_keys:
        img = steps.get(k)
        if img is not None:
            panels.append((k, img))

    if not panels:
        raise ValueError("No images to display.")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, img) in zip(axes, panels):
        ax.imshow(_to_rgb(img))
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Quick demo helpers for a notebook
# ---------------------------------------------------------------------------

def demo_from_image_path(img_path: str, params_path: str = "vision_params.json"):
    """
    Load an image from disk, run the pipeline, and show intermediate results.
    """
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    frame_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    params = load_params(params_path)
    steps = run_pipeline(frame_bgr, params)
    show_steps(steps, title=f"Pipeline demo: {Path(img_path).name}")
    return steps


def demo_from_camera(params_path: str = "vision_params.json", device_index: Optional[int] = None):
    """
    Capture one frame from camera, run the pipeline, and show intermediate results.
    """
    params = load_params(params_path)
    frame = grab_single_frame(device_index=device_index)
    steps = run_pipeline(frame, params)
    show_steps(steps, title="Pipeline demo (live camera frame)")
    return steps


if __name__ == "__main__":
    # Minimal CLI preview (saves nothing, just runs if a camera is available)
    try:
        display_report()
        print("\nRunning quick camera demo...")
        demo_from_camera()
        plt.show()
    except Exception as exc:
        print(f"[report_jules_vision] Skipped live demo: {exc}")
