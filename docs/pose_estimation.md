# State estimation

**TODO:** Add justification why EKF

## Extended Kalman Filter (EKF)

As described in the [project description document](../docs/project_description.md), the state-space equations are nonlinear due to the projections of the foward and angular velocities of the Thymio. Therefore, we have to use the Extended Kalman Filter to linearized the system at each discrete time k.

Let's consider the previously defined model of the system but with some Gaussian nois incorporated into it:

```math
\begin{equation}
    \begin{align*}
        \vec{x}_{k} &= \boldsymbol{g}(\vec{x}_{k-1}, \vec{u}_k) + \epsilon_t = 
            \begin{pmatrix}
                x_{k-1} + t_s \cdot v_{k-1} \cdot cos(\theta_{k-1})\\ 
                y_{k-1} + t_s \cdot v_{k-1} \cdot sin(\theta_{k-1})\\
                \theta_{k-1} - t_s \cdot \omega_{k-1}\\
                \frac{\lambda}{2} (u_{rk} + u_{lk})\\
                \frac{\lambda}{2l} (u_{rk} - u_{lk})
            \end{pmatrix} + \epsilon_t \quad \text{with} \quad \vec{u}_k =
                \begin{pmatrix}
                    u_{rk} \\
                    u_{lk}
                \end{pmatrix} \\
        \vec{z}_k &= \boldsymbol{h}(\vec{x}_{k}) + \delta_t =
            \begin{pmatrix}
                x_k \\
                y_k \\
                \theta_k \\
                \frac{1}{\lambda}(v_k + l\omega_k)\\
                \frac{1}{\lambda}(v_k - l\omega_k)
            \end{pmatrix} + \delta_t
    \end{align*}
\end{equation}
```

Where $\epsilon_t \sim N(0, Q)$ is a multinormal modelling the uncertainty introduced by the state-space transition, and $\delta_t \sim N(0, R)$ is a multinormal modelling the measurement noise.

### Prediction step

1. We need to estimate the covariance $Q$ of the stochastic noise $\epsilon_t \sim N(0, Q)$ affecting the state-space model.

```math
\begin{equation}
    Q = 
    \begin{pmatrix}
        q_x & 0 & 0 & 0 & 0 \\
        0 & q_y & 0 & 0 & 0 \\
        0 & 0 & q_\theta & 0 & 0 \\
        0 & 0 & 0 & q_v & 0 \\
        0 & 0 & 0 & 0 & q_\omega
    \end{pmatrix} = ?
\end{equation}
```

2. We have to compute the jacobian of the motion model $\boldsymbol{g}(\vec{x}_{k-1}, \vec{u}_k)$, denoted $G$:

```math
\begin{equation}
    \begin{align*}
        G_k &= 
        \begin{pmatrix}
            \frac{\partial g_1}{\partial x_{k-1}} & \frac{\partial g_1}{\partial y_{k-1}} & \frac{\partial g_1}{\partial \theta_{k-1}} & \frac{\partial g_1}{\partial v_{k-1}} & \frac{\partial g_1}{\partial \omega_{k-1}} \\
            \frac{\partial g_2}{\partial x_{k-1}} & \frac{\partial g_2}{\partial y_{k-1}} & \frac{\partial g_2}{\partial \theta_{k-1}} & \frac{\partial g_2}{\partial v_{k-1}} & \frac{\partial g_2}{\partial \omega_{k-1}} \\
            \frac{\partial g_3}{\partial x_{k-1}} & \frac{\partial g_3}{\partial y_{k-1}} & \frac{\partial g_3}{\partial \theta_{k-1}} & \frac{\partial g_3}{\partial v_{k-1}} & \frac{\partial g_3}{\partial \omega_{k-1}} \\
            \frac{\partial g_4}{\partial x_{k-1}} & \frac{\partial g_4}{\partial y_{k-1}} & \frac{\partial g_4}{\partial \theta_{k-1}} & \frac{\partial g_4}{\partial v_{k-1}} & \frac{\partial g_4}{\partial \omega_{k-1}} \\
            \frac{\partial g_5}{\partial x_{k-1}} & \frac{\partial g_5}{\partial y_{k-1}} & \frac{\partial g_5}{\partial \theta_{k-1}} & \frac{\partial g_5}{\partial v_{k-1}} & \frac{\partial g_5}{\partial \omega_{k-1}}
        \end{pmatrix} \\
        &=
        \begin{pmatrix}
            1 & 0 & -t_sv_{k-1}sin(\theta_{k-1}) & t_scos(\theta_{k-1}) & 0 \\
            0 & 1 & t_sv_{k-1}cos(\theta_{k-1}) & t_ssin(\theta_{k-1}) & 0 \\
            0 & 0 & 1 & 0 & -t_s \\
            0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0
        \end{pmatrix}
    \end{align*}
\end{equation}
```

### Measurement update step

1. We need to estimate the covariance $R$ of the stochastic noise $\delta_t \sim N(0, R)$ describing the measurement noise.

```math
\begin{equation}
    R =
    \begin{pmatrix}
        r_x & 0 & 0 & 0 & 0 \\
        0 & r_y & 0 & 0 & 0 \\
        0 & 0 & r_\theta & 0 & 0 \\
        0 & 0 & 0 & r_{u_r} & 0 \\
        0 & 0 & 0 & 0 & r_{u_l}
    \end{pmatrix} = ?
\end{equation}
```

```math
\begin{equation}
    R_{reduced} =
    \begin{pmatrix}
        0 & 0 & 0 & r_{u_r} & 0 \\
        0 & 0 & 0 & 0 & r_{u_l}
    \end{pmatrix} = ?
\end{equation}
```

2. We have to compute the jacobian of the measurement model $\boldsymbol{h}(\vec{x}_{k})$, denoted $H$:

```math
\begin{equation}
    \begin{align*}
        H_k &= 
        \begin{pmatrix}
            \frac{\partial h_1}{\partial x_k} & \frac{\partial h_1}{\partial y_k} & \frac{\partial h_1}{\partial \theta_k} & \frac{\partial h_1}{\partial v_k} & \frac{\partial h_1}{\partial \omega_k} \\
            \frac{\partial h_2}{\partial x_k} & \frac{\partial h_2}{\partial y_k} & \frac{\partial h_2}{\partial \theta_k} & \frac{\partial h_2}{\partial v_k} & \frac{\partial h_2}{\partial \omega_k} \\
            \frac{\partial h_3}{\partial x_k} & \frac{\partial h_3}{\partial y_k} & \frac{\partial h_3}{\partial \theta_k} & \frac{\partial h_3}{\partial v_k} & \frac{\partial h_3}{\partial \omega_k} \\
            \frac{\partial h_4}{\partial x_k} & \frac{\partial h_4}{\partial y_k} & \frac{\partial h_4}{\partial \theta_k} & \frac{\partial h_4}{\partial v_k} & \frac{\partial h_4}{\partial \omega_k} \\
            \frac{\partial h_5}{\partial x_k} & \frac{\partial h_5}{\partial y_k} & \frac{\partial h_5}{\partial \theta_k} & \frac{\partial h_5}{\partial v_k} & \frac{\partial h_5}{\partial \omega_k} \\
        \end{pmatrix} \\
        &=
        \begin{pmatrix}
            1 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & \frac{1}{\lambda} & \frac{l}{\lambda} \\
            0 & 0 & 0 & \frac{1}{\lambda} & -\frac{l}{\lambda}
        \end{pmatrix}
    \end{align*}
\end{equation}
```

```math
\begin{equation}
    \begin{align*}
        H_{k,reduced} &= 
        \begin{pmatrix}
            0 & 0 & 0 & \frac{1}{\lambda} & \frac{l}{\lambda} \\
            0 & 0 & 0 & \frac{1}{\lambda} & -\frac{l}{\lambda}
        \end{pmatrix}
    \end{align*}
\end{equation}
```

**TODO:**
- Add the feature that there is two $H_k$ possible, one used when the cam is available and one to use when there is no info available from the cam

- Compute $Q$ and $R$ based on the exercise 8, so maybe tweek it a bit to have the correct estimation of the covariance matrix. Use the assumption that 50% of error is due to measure and 50% due to the model or make a more complex assumption if there is an intuition to do so.

- At each main iteration, just before motion control, call Kalman and use the $\mu$ for the control

- OPTIONAL: Optimize the Kalman with 4/5 iterations only odometry and 1/5 with camera