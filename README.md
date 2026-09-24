# 2D MR-LBM CUDA Solver

A 2D lattice Boltzmann simulation project implemented with **C++ and CUDA**, featuring a **Moment Representation Lattice Boltzmann Method (MR-LBM)** solver and real-time visualization with **OpenGL**.

## Features

* 2D **D2Q9 MR-LBM** solver
* Moment-based representation:

  $$
  (\rho, u_x, u_y, \Pi_{xx}, \Pi_{yy}, \Pi_{xy})
  $$
* Distribution reconstruction and **pull streaming**
* Moment-space **BGK collision** with force correction
* CUDA-accelerated GPU computation
* Dynamic boundary handling
* Kármán vortex flow around a moving circular obstacle
* Jet flow simulation
* Real-time OpenGL visualization

## Solver Pipeline

The main simulation loop follows:

```text
Moment Storage
      ↓
D2Q9 Distribution Reconstruction
      ↓
Pull Streaming
      ↓
Moment Recovery
      ↓
Moment-Space Collision
      ↓
Update
```

## Implementation

The solver stores six macroscopic/moment variables at each grid cell instead of directly storing all D2Q9 distribution functions. CUDA kernels are used to accelerate the main LBM computation on the GPU, while the CPU handles case setup, boundary updates, and visualization data preparation.

The project also includes CUDA memory management and Host–Device data synchronization for dynamic simulations.

## Demo Cases

### Kármán Vortex

Flow past a circular obstacle to demonstrate vortex shedding and dynamic boundary handling.

### Jet Flow

A jet entering the computational domain through a centered inlet.

## Tech Stack

* **C++**
* **CUDA**
* **OpenGL / FreeGLUT**
* **D2Q9 LBM**
* **MR-LBM**

## Project Goal

The project was developed to study the implementation details of **moment-representation LBM**, GPU acceleration with CUDA, and the interaction between numerical fluid simulation and real-time visualization.

## Runtime effect
<img width="1012" height="753" alt="屏幕截图 2026-09-25 002959" src="https://github.com/user-attachments/assets/3ebf97a8-c584-4187-b30f-67689d4f3985" />
<img width="1016" height="751" alt="屏幕截图 2026-09-25 003023" src="https://github.com/user-attachments/assets/b483921e-3cb7-4c97-b53b-33cff6b79ade" />
<img width="1012" height="748" alt="屏幕截图 2026-09-25 003100" src="https://github.com/user-attachments/assets/02da0436-0a12-49c6-8531-a21f48205c00" />
<img width="1021" height="755" alt="屏幕截图 2026-09-25 003115" src="https://github.com/user-attachments/assets/59024e75-87e2-4f31-ae02-8888438427b9" />
