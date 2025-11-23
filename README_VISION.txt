Guide rapide vision (version simplifiée)
========================================

Pipeline (une seule initialisation, pas de warp/crop, pas de touches r/p) :
1) Init couleurs robot : trackbars Hmin/Hmax rouge/vert + aires min/max avec aperçu masques.
2) Blur robot : suppression automatique du robot (masque dilaté proportionnel à sa taille).
3) Init Canny : trackbars low/high avec aperçu des edges (pas de polylines).
4) Init polygones : trackbars min/max area avec aperçu des obstacles sur l'image (robot déjà blur).
5) Polygones figés : calculés une seule fois à l'init, réutilisés ensuite en live.
6) Détection continue : robot + obstacles (obstacles = ceux figés à l'init).

Fichier principal : src/computer_vision/vision.py

Fonctions principales
- initialize(frame) : lance les étapes 1→4, fige les polygones une fois.
- process(frame, show_debug=False) : retourne (polys, robot_state[, debug_img]) avec les paramètres figés.
- init_colors / init_canny / init_polygons : étapes manuelles avec trackbars.
- blur_robot : inpaint proportionnel à la taille du robot pour éviter de l’inclure dans les obstacles.
- Lissage : la pose robot (centre/theta) est lissée (smooth_center/smooth_theta).

Mode démo (voir __main__ dans vision.py)
- Image fixe : charge une image (relative au dossier images/) puis initialize() et process().
- Webcam : lit la première frame, initialize(), puis boucle de détection (q/ESC pour quitter si show_debug=True).
- show_debug=True pour afficher le retour vision (sinon les données peuvent être consommées par un autre module).

Réglages utiles
- Tolérance robot : dans robot_pair_ok/score_pair (dist_factor, max_r_rel_diff) et robot_area min/max.
- Canny : paramètres low/high via init_canny ; renforcer/adoucir la fermeture edges dans strengthen_edges (ksize, iterations).
- Polygones : min_area / max_area via init_polygons.

Ce qui a été retiré
- Warp/crop manuel ou auto, recalibrations en boucle. Objectif : code court, init unique, polygones figés.
