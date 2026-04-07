# BlockTower Dataset

This README documents only the raw dataset files. It describes the contents and format of the four zip archives below, each of which extracts into a flat folder of raw `.npy` files. Code-level loading behavior and processed tensors will be documented in a separate README.

This dataset contains four zip files storing rigid-body physical trajectories of a 3D block tower environment, simulated using Blender:

  - `toy_blocks1-10.zip`
  - `main_blocks17-31.zip`
  - `euler_toy_blocks1-10.zip`
  - `euler_main_blocks17-31.zip`

In toy datasets, all scenes only contain blocktowers with blocks ranging from 1 to 10 while in main datasets, all scenes contain blocktowers with blocks ranging from 17 to 31.

The prefix euler means rotation is represented by euler angle instead of quaternion.

## Ground Object Specifications

The specifications below describe the raw `.npy` files only.

For all zip files, the ground acts as the passive collision floor and is always placed at index `0` of the object list.

* **Position**: `[0.0, 0.0, -0.5]` (Center is offset downwards so its top surface rests perfectly at $Z=0$).
* **Size**: `[100.0, 100.0, 1.0]` (A massive box to catch all blocks).
* **Velocities & Rotations**: Strictly `0.0`.
* **Mask**: `0.0` (Neural network models should use this mask to zero out gradient updates for the ground).

## Toy_blocks1-10 and main_blocks17-31

### Data Overview

* **File Format**: `.npy` (NumPy Array)
* **Data Shape**: `[Frames, Object_Number, Feature_Dimension]`
  * `Frames`: Fixed to `150` frames per file.
  * `Object_Number`: Variable, depends on the scene. Typically $N+1$ (where $N$ is the number of blocks, and `1` is the ground). The **Ground is ALWAYS index 0**.
  * `Feature_Dimension`: `17`
* **Simulation Framerate (FPS)**: `25`
* **Time Step ($dt$)**: `0.04` seconds per frame.

### Feature Dimension Breakdown (17-Dim)

Each object at any given frame is represented by a 17-dimensional feature vector:

| Index | Name | Notation | Unit | Description |
| :---: | :--- | :---: | :---: | :--- |
| `0` | Position X | $x$ | Meter | Global X coordinate of the object's center. |
| `1` | Position Y | $y$ | Meter | Global Y coordinate of the object's center. |
| `2` | Position Z | $z$ | Meter | Global Z coordinate of the object's center. |
| `3` | Quaternion X | $q_x$ | Unitless | X-component of unit quaternion (vector part). **(xyzw order / ‖q‖=1)** |
| `4` | Quaternion Y | $q_y$ | Unitless | Y-component of unit quaternion (vector part). **(xyzw order / ‖q‖=1)** |
| `5` | Quaternion Z | $q_z$ | Unitless | Z-component of unit quaternion (vector part). **(xyzw order / ‖q‖=1)** |
| `6` | Quaternion W | $q_w$ | Unitless | W-component of unit quaternion (scalar part). **(xyzw order / ‖q‖=1)** |
| `7` | Length X | $l_x$ | Meter | Full extent (size) of the object along its local X-axis. |
| `8` | Length Y | $l_y$ | Meter | Full extent (size) of the object along its local Y-axis. |
| `9` | Length Z | $l_z$ | Meter | Full extent (size) of the object along its local Z-axis. |
| `10` | Dynamic Mask | $m$ | Boolean | `1.0` for active dynamic blocks, `0.0` for static ground. |
| `11` | Linear Vel X | $v_x$ | m/s | Linear velocity along global X-axis. |
| `12` | Linear Vel Y | $v_y$ | m/s | Linear velocity along global Y-axis. |
| `13` | Linear Vel Z | $v_z$ | m/s | Linear velocity along global Z-axis. |
| `14` | Angular Vel X | $\omega_x$ | rad/s | Angular velocity around global X-axis. |
| `15` | Angular Vel Y | $\omega_y$ | rad/s | Angular velocity around global Y-axis. |
| `16` | Angular Vel Z | $\omega_z$ | rad/s | Angular velocity around global Z-axis. |

## Euler_toy_blocks1-10 and euler_main_blocks17-31

### Data Overview

* **File Format**: `.npy` (NumPy Array)
* **Data Shape**: `[Frames, Object_Number, Feature_Dimension]`
  * `Frames`: Fixed to `150` frames per file.
  * `Object_Number`: Variable, depends on the scene. Typically $N+1$ (where $N$ is the number of blocks, and `1` is the ground). The **Ground is ALWAYS index 0**.
  * `Feature_Dimension`: `16`
* **Simulation Framerate (FPS)**: `25`
* **Time Step ($dt$)**: `0.04` seconds per frame.

### Feature Dimension Breakdown (16-Dim)

Each object at any given frame is represented by a 16-dimensional feature vector:

| Index | Name | Notation | Unit | Description |
| :---: | :--- | :---: | :---: | :--- |
| `0` | Position X | $x$ | Meter | Global X coordinate of the object's center. |
| `1` | Position Y | $y$ | Meter | Global Y coordinate of the object's center. |
| `2` | Position Z | $z$ | Meter | Global Z coordinate of the object's center. |
| `3` | Euler X | $r_x$ | Radian | Rotation around X-axis. **(Continuous / Unwrapped)** |
| `4` | Euler Y | $r_y$ | Radian | Rotation around Y-axis. **(Continuous / Unwrapped)** |
| `5` | Euler Z | $r_z$ | Radian | Rotation around Z-axis. **(Continuous / Unwrapped)** |
| `6` | Length X | $l_x$ | Meter | Full extent (size) of the object along its local X-axis. |
| `7` | Length Y | $l_y$ | Meter | Full extent (size) of the object along its local Y-axis. |
| `8` | Length Z | $l_z$ | Meter | Full extent (size) of the object along its local Z-axis. |
| `9` | Dynamic Mask | $m$ | Boolean | `1.0` for active dynamic blocks, `0.0` for static ground. |
| `10` | Linear Vel X | $v_x$ | m/s | Linear velocity along global X-axis. |
| `11` | Linear Vel Y | $v_y$ | m/s | Linear velocity along global Y-axis. |
| `12` | Linear Vel Z | $v_z$ | m/s | Linear velocity along global Z-axis. |
| `13` | Angular Vel X | $\omega_x$ | rad/s | Angular velocity around global X-axis. |
| `14` | Angular Vel Y | $\omega_y$ | rad/s | Angular velocity around global Y-axis. |
| `15` | Angular Vel Z | $\omega_z$ | rad/s | Angular velocity around global Z-axis. |

### Coordinate System & Conventions

* **World Coordinates**: Blender standard **Right-Handed Z-Up** system.
  * `+X`: Right
  * `+Y`: Forward/Depth
  * `+Z`: Up (Gravity acts along `-Z`)
* **Euler Angles Order**: **XYZ** (Extrinsic). The rotation is applied sequentially around the global X, then Y, then Z axes.
* **Continuous Euler Angles (Unwrapped)**:
  To prevent singularity jumps (e.g., jumping from $\pi$ back to $-\pi$) and resulting angular velocity spikes, the Euler angles in this dataset are **continuous**. If an object rotates multiple times, its angle will accumulate beyond $2\pi$ or drop below $-2\pi$ (e.g., $10\pi$ for 5 full rotations).
* **Velocity Calculation**: Both linear and angular velocities are calculated via **first-order finite differences** from the positional and unwrapped Euler angle trajectories over $dt = 1/25 = 0.04s$.
