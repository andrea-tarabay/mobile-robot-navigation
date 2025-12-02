import numpy as np
import matplotlib.pyplot as plt
import math

def wrap_to_pi(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def plot_robot_local(x, y, theta, goal):
    """
    Repère :
      - x → droite
      - y → bas
      - theta CCW depuis +x
      - y_forward = devant robot
      - x_left    = gauche robot
    """

    xg, yg = goal
    dx = xg - x
    dy = yg - y

    # 1) angle global vers le goal
    angle_goal = math.atan2(dy, dx)
    print(f"angle_goal (atan2) = {angle_goal:.3f} rad")
    angle_goal = wrap_to_pi(angle_goal)
    print(f"wrap_to_pi (atan2) = {angle_goal:.3f} rad")


    # 2) erreur angulaire alpha en repère robot
    y_forward =  math.cos(theta) * dx + math.sin(theta) * dy
    x_left    = -math.sin(theta) * dx + math.cos(theta) * dy
    alpha = math.atan2(x_left, y_forward)
    alpha = wrap_to_pi(alpha)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.invert_yaxis()  # IMPORTANT : y vers le bas comme ta caméra
    ax.grid(True)

    # --- Robot ---
    ax.plot(x, y, "ko")  # robot position

    # --- Flèche theta ---
    ax.arrow(x, y,
             40 * math.cos(theta),
             40 * math.sin(theta),
             head_width=10, color="blue",
             length_includes_head=True)

    # --- Flèche direction goal ---
    ax.arrow(x, y,
             40 * math.cos(angle_goal),
             40 * math.sin(angle_goal),
             head_width=10, color="green",
             length_includes_head=True)

    # --- Flèche alpha (repère robot) ---
    norm = max(1e-6, math.hypot(x_left, y_forward))
    ax.arrow(x, y,
             40 * (x_left / norm),
             40 * (y_forward / norm),
             head_width=10, color="red",
             length_includes_head=True)

    # --- Goal ---
    ax.plot(xg, yg, "rx", markersize=12)

    # --- Affichage des valeurs numériques ---
    text = (
        f"theta = {theta:.3f} rad\n"
        f"angle_goal (atan2) = {angle_goal:.3f} rad\n"
        f"alpha = {alpha:.3f} rad"
    )

    ax.text(x + 60, y - 60, text,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7))

    plt.show()

# Exemple :
plot_robot_local(
    x=150, y=200,
    theta=math.radians(30),
    goal=(400, 350)
)