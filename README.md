# 🦾 Mobile Robotics Project

**Course:** Basics of Mobile Robotics — Prof. Francesco Mondada <br>
**Institution:** EPFL <br>
**Semester:** Fall 2025 <br>

---

## 📘 Project Overview

This project focuses on the **design, implementation, and demonstration** of an autonomous mobile robot (Thymio) capable of navigating a known environment, planning optimal paths, and reacting to unexpected obstacles using both **global** and **local navigation** techniques.

The project aims to integrate multiple modules of mobile robotics, including:

* **Environment mapping using computer vision**
* **Path planning with global navigation**
* **Local obstacle avoidance**
* **Motion control**
* **Pose estimation through Bayesian filtering**

---

## 🧭 Project Objectives

1. **Create a Simulated Environment**

   * Define an environment with obstacles the robot must avoid using *global navigation* (not relying on onboard sensors).

2. **Path Planning**

   * Implement an algorithm that enables the robot to find the optimal path from any arbitrary starting position to a target point.

3. **Motion Control & Pose Estimation**

   * Design a motion control system allowing the robot to follow the planned path accurately.
   * Estimate the robot’s position using **Bayesian filtering** (e.g., Kalman or Particle Filter).

4. **Local Obstacle Avoidance**

   * Integrate local navigation strategies to handle unforeseen physical obstacles in real time.

---

## 🧩 Project Deliverables

### 1. 📓 Jupyter Notebook Report

A comprehensive Jupyter Notebook containing:

* Group members and their responsibilities
* Introduction and description of the environment
* Theoretical background and design decisions
* Implementation details and modular code snippets
* Visualizations (maps, trajectories, pose estimations)
* Final results and conclusions

> 💡 The notebook should be structured, well-documented, and cite all sources appropriately.

### 2. 💻 Executable Code

* Clean, modular Python code for each subsystem (path planning, control, filtering, etc.).
* A main script or notebook section that runs the full system.

### 3. 🎥 Live Demonstration

* **3-minute presentation** introducing the approach and design.
* **3-minute live demo** showing real-time mapping, navigation, and pose estimation.
* **Q&A session** with instructors.
* Include a **backup video** of the system in case of technical issues.

---

## 🗓️ Deadlines

**Final Submission Deadline:** December 4th, 23:00
**Presentation:** 3 min intro + 3 min demo + 20 min Q&A session

---

## 🏁 Expected Outcome

By the end of this project, your robot should:

* Successfully navigate from a start to a goal position.
* Handle unexpected obstacles dynamically.
* Display accurate pose estimation even under disturbances (e.g., “kidnapping” scenarios).

---

