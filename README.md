# 3D_Reconstruction_with_Gaussian_Splatting_using Depth Anything 3

This project implements a pipeline for monocular depth estimation and 3D reconstruction using the **Depth Anything 3 (DA3)** model. It processes images or video frames to generate high-quality depth maps, confidence maps, and 3D assets (GLB and Gaussian Splatting PLY).

## Installation

### 1. Install Depth Anything 3
This project is built upon the **Depth Anything 3** library. You must install it from the source before running the pipeline:

[https://github.com/ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)

### 2. Install Python Dependencies
Ensure you have a Python environment with `torch` and CUDA support enabled. Install the remaining dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

## Project Structure

* **`main.ipynb`**: The primary orchestration notebook. It defines the end-to-end execution pipeline, managing configuration, model initialization, data loading, inference execution, and the final export of 3D reconstruction assets.

* **`functions.py`**: The core logic module encapsulating heavy-lifting operations. It contains routines for model loading (DepthAnything3), data ingestion (image scanning and video frame extraction), 3D point cloud post-processing (statistical outlier removal via scipy), and wrapper functions for executing inference and exporting GLB/Gaussian Splatting files.

* **`helper_functions.py`**: A utility suite focused on I/O and visualization. It handles file system management (directory setup) and provides rendering tools for 2D depth/confidence maps (using matplotlib) and interactive 3D point cloud visualization (using Open3D).

## Implementation Details

### 1. Model Initialization

The project uses the `DepthAnything3` API. By default, it loads the large nested model variant to ensure high-quality depth estimation:

* **Model**: `depth-anything/DA3NESTED-GIANT-LARGE`
* **Device**: Automatically selects `cuda` if available, falling back to `cpu`.

### 2. Data Loading

The pipeline supports processing both static images and video files:

* **Images**: Scans a specified folder for common image formats (`.png`, `.jpg`, `.bmp`, etc.).
* **Video**: Includes functionality to extract individual frames from a video file at a specified frame rate for processing.

### 3. Inference & Depth Estimation

The core inference step calculates several key metrics for each input view:

* **Depth Maps**: Per-pixel depth information.
* **Confidence Maps**: Estimates the reliability of the depth prediction.
* **Camera Parameters**: Infers extrinsics and intrinsics for the scene.
* **Gaussian Splatting Inference**: `infer_gs=True` is enabled during inference to prepare for 3D reconstruction.

### 4. Visualization

Utilities provided in `da3_utils.py` allow for immediate inspection of results:

* **2D Visualization**: Side-by-side comparison of the RGB Image, Depth Map (`turbo` colormap), and Confidence Map (`viridis` colormap).
* **3D Visualization**: Interactive point cloud visualization using Open3D.

### 5. Exporting Results

The final step exports the processed data into standard 3D formats for use in other applications:

* **Formats**: Exports to **GLB** (with cameras) and **Gaussian Splatting PLY**.
* **Filtering**: Utilizes a confidence threshold (default: 40th percentile) to filter out low-confidence points.
* **Point Limit**: Caps the output at 10 million points to ensure manageability.


## Usage

1. **Setup Environment**: Install dependencies and the DA3 library as linked above.
2. **Prepare Data**: Place your images or video in a data folder (e.g., `./data/room_photos`).
3. **Run Notebook**: Open `main.ipynb` and execute the cells in order.
* The `setup_paths` function will automatically create necessary `results` and `masks` directories.
* The final steps will export your 3D models to the results directory.
