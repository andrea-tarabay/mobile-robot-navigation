Guide d'utilisation - Vision (détection robot + obstacles)
===========================================================

Ce fichier résume comment utiliser la pipeline vision pour les autres parties du projet (global path, local path, Kalman, etc.).

1) Calibration initiale (une seule fois ou quand l'éclairage change)
- Lancer la webcam et récupérer une frame d’init : `vision = Vision()` puis `vision.run_webcam(cam_index=0, params_path=vision.default_params_path)`.
- Si aucun JSON de paramètres n’existe, les trackbars s’ouvrent pour `init_colors` (HSV) puis `init_canny`. À la fin, les seuils sont sauvegardés dans `src/computer_vision/vision_params.json`.
- Si le fichier existe, il est rechargé automatiquement (pas de trackbars).
- La carte des obstacles est figée sur la première frame via `freeze_map_from_frame`, pour séparer une bonne fois la map et le robot.

Touches dans la fenêtre live :
- `r` : refreeze la map statique si la scène a bougé (clear + freeze sur la frame courante).
- `p` : recalibration complète (trackbars) + sauvegarde; refreeze de la map derrière.
- `q` ou `ESC` : quitter proprement.

Affichage live :
- Obstacles en vert (polygones Canny, tirés du cache si map figée).
- Robot brut (blanc + flèche rouge).
- Robot lissé (jaune) = sortie EMA, à utiliser pour le planner/Kalman.
- Ligne “miss=XX%” : ratio de frames où le robot n’a pas été trouvé; un warning console apparaît si ce ratio dépasse `warn_threshold` (par défaut 25 %).

2) Intégration dans une boucle temps réel
- Initialiser et geler la map une fois (au setup) :
  ```python
  vision = Vision()
  vision.load_params()             # ou trackbars si fichier absent
  ret, frame0 = cap.read()         # une frame webcam au démarrage
  polys = vision.freeze_map_from_frame(frame0)
  ```
  `polys` fournit les obstacles pour le path planning (buffer, graph, etc.).

- Dans la boucle :
  ```python
  ret, frame = cap.read()
  robot = vision.detect_robot(frame)   # la map reste en cache
  if robot["found"]:
      pose_center = robot["smooth_center"]
      pose_theta  = robot["smooth_theta"]
      # envoyer pose lissée au Kalman/local path
  ```
  Ne recalculer la map que si le décor change : `vision.clear_static_map(); polys = vision.freeze_map_from_frame(frame)`.

3) Fonctions utiles
- `detect_robot(frame)`: renvoie un dict (pose brute + lissée + masque). Paramètre `log_stats=False` pour ne pas impacter le monitoring.
- `process(frame)`: renvoie la liste des polygones obstacles; s’appuie sur le cache si `freeze_map_from_frame` a été appelé.
- `freeze_map_from_frame(frame)`: calcule et fige la map statique (obstacles) pour la session.
- `clear_static_map()`: force un recalcul de la map sur la prochaine frame.
- `save_params(filepath) / load_params(filepath)`: sauvegarde/recharge les seuils HSV+Canny (JSON). Par défaut `src/computer_vision/vision_params.json`.
- `run_webcam(cam_index=0, params_path=..., warn_ratio=...)`: mode de test tout-en-un avec affichage.

4) Paramétrage
- `pose_filter_alpha` (EMA) : 0.2–0.4 typiquement (plus haut = plus réactif, plus bas = plus lisse).
- `warn_threshold` (dans `not_found_stats` ou via `warn_ratio` de `run_webcam`) : seuil d’alerte pour frames manquées.

5) Exemple minimal hors webcam
```python
from computer_vision.vision import Vision
vision = Vision()
img = vision.load("table7.jpg")
vision.load_params()         # ou vision.init_colors/img + vision.init_canny(img)
polys = vision.freeze_map_from_frame(img)
robot = vision.detect_robot(img)
print(robot["smooth_center"], robot["smooth_theta"])
```
