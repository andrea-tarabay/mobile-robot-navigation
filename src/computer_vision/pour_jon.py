"""
Petit exemple minimal pour Jon : comment brancher la vision et récupérer
l'historique (x, y, theta) pour son code de contrôle/filtre.

Usage :
    python -m src.computer_vision.pour_jon

Points clés :
- Une mesure est produite à chaque appel de vision.process(...) quand log_pose=True.
- La fréquence réelle des mesures = cadence d'images traitées (webcam + temps CPU),
  typiquement entre ~15 et ~30 Hz selon la machine et la résolution. Mesure-la dans
  ta boucle si tu as besoin d'un dt précis (exemple ci-dessous).
"""

import time
import cv2
import numpy as np

from .vision import Vision


def main():
    # Init vision sur la première frame (trackbars).
    vision = Vision()
    show_debug = False  # laisse False si ton code consomme juste les données

    cap = cv2.VideoCapture(0)
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Impossible de lire la webcam.")
    vision.initialize(frame0)

    # Boucle exemple : on traite 200 frames puis on récupère l'historique.
    fps_smoothed = None
    for _ in range(200):
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break

        polys, st = vision.process(frame, show_debug=show_debug)

        # Exemple d'exploitation directe (pose instantanée)
        if st["found"]:
            cx, cy = st["smooth_center"] or st["center"]
            theta = st["smooth_theta"] or st["theta"]
            # Ici tu peux commander ton robot avec (cx, cy, theta)
            # control_robot(cx, cy, theta)

        # Estimation instantanée de la fréquence pour info/log
        dt = time.time() - t0
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        if fps_smoothed is None:
            fps_smoothed = inst_fps
        else:
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * inst_fps
        print(f"FPS (vision) ~ {fps_smoothed:5.1f}", end="\r")

    cap.release()
    cv2.destroyAllWindows()
    print()  # retour à la ligne après la barre \r

    # Récupération des mesures pour le filtre de Kalman
    poses = vision.get_pose_array(as_numpy=True, clear=True)  # shape (N, 3)
    print(f"{len(poses)} poses collectées, exemple: {poses[:3]}")

    # Sauvegarde optionnelle sur disque
    np.save("poses_demo.npy", poses)
    print("Sauvegardé dans poses_demo.npy (x, y, theta) en float32.")


if __name__ == "__main__":
    main()
