# Determining-Material-Parameters-using-Physics-Informed-Neural-Networks

This repository contains the code used in the master thesis:

Physics-Informed Neural Networks for Non-Invasive Estimation of Valve Tissue Elastic Parameters: Framework Adaptation and Literature Review
Jonas Wikse, NTNU, 2026

The thesis investigates the use of inverse Physics-Informed Neural Networks (PINNs) for non-invasive estimation of elastic material parameters in mitral valve tissue. The work is based on adapting and verifying PINN frameworks for inverse elasticity problems, with particular focus on Young's modulus (E) and Poisson's ratio (\nu).

Repository overview

The repository contains code for the main verification attempts and numerical experiments presented in the thesis.

.
├── 2D Thick-walled Cylinder/
│   └── Code and data for Example Code 1
│
├── 3D_Linear_Elasticity/
│   └── Code and data for Example Code 3
│
├── Ex4_Neo-Hookean.py
│   └── Neo-Hookean PINN implementation for patient-specific data testing
│
└── README.md

<img width="529" height="232" alt="image" src="https://github.com/user-attachments/assets/2a4f90c0-e401-441d-9680-9364cf212ae6" />


Main packages include:

deepxde
torch
numpy
matplotlib
scipy

Additional software used in the thesis workflow:

3D Slicer
Rhino
MATLAB
Abaqus

Jonas Wikse
Department of Structural Engineering
Norwegian University of Science and Technology
NTNU, 2026

Disclaimer

This repository is intended for research and educational purposes only. The code is not intended for clinical use.
