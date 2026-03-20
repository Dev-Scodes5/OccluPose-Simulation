# OccluPose: Biomechanical Trajectory Inference

This repository contains the simulation code and mathematical proofs for **OccluPose**, a framework designed to recover joint trajectories during severe visual occlusion in athletic environments. 

The core of this project is a spatiotemporal inference engine that uses **Gaussian Process (GP) Regression** to bridge gaps in computer vision data. Instead of relying on purely spatial heatmaps (which fail when a limb is hidden), OccluPose treats movement as a continuous probabilistic function bounded by Newtonian physics.

## The Problem: "Limb Collapse" in HPE
Standard Human Pose Estimation (HPE) models like HRNet or OpenPose treat every video frame as a discrete event. When an athlete's limb is blocked by a hurdle or equipment, these models "lose" the joint, leading to jitter or physically impossible coordinate jumps (limb collapse).

OccluPose solves this by formulating the recovery as a **Maximum A Posteriori (MAP)** optimization problem, using temporal momentum to maintain kinematic fidelity where visual data is zero.

## Simulation: The Biomechanical Proxy
Due to the proprietary nature of high-end athletic tracking data, this repository utilizes a **Biomechanical Proxy Simulation** to validate the math.

- **Trajectory:** A 3D Newtonian reconstruction of a right-knee hurdle jump (60 FPS).
- **Occlusion:** A 20-frame "blind window" simulating a total loss of visual signal.
- **Baseline:** An HRNet-W32 architecture simulation with spatial noise diffusion.

### Quick Start
This script requires `numpy`, `matplotlib`, and `scikit-learn`.

```bash
# Clone the repository
git clone [https://github.com/Dev-Scodes5/OccluPose-Simulation.git](https://github.com/Dev-Scodes5/OccluPose-Simulation.git)

# Run the inference simulation
python occlupose_proxy.py
