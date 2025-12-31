# 3D_Reconstruction_with_Gaussian_Splatting

# Depth Anything 3: Depth Estimation & 3D Export Project

This project implements a pipeline for monocular depth estimation and 3D reconstruction using the **Depth Anything 3 (DA3)** model. It processes images or video frames to generate high-quality depth maps, confidence maps, and 3D assets (GLB and Gaussian Splatting PLY).

## ⚠️ Prerequisite: Installation

**Before running this project, you must install the Depth Anything 3 library.**

Please follow the official installation instructions provided in the ByteDance repository:
**[https://github.com/ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)**

Ensure you have a working environment with `torch` and CUDA support enabled.

## Project Structure

* **`main.ipynb`**: The main Jupyter Notebook that orchestrates the entire pipeline, from loading data to exporting 3D models.
* **`da3_utils.py`**: A utility module containing helper functions for file management, directory setup, and visualization of results.

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

## Dependencies

In addition to the `depth_anything_3` library, this project requires:

* `numpy`
* `matplotlib`
* `opencv-python` (`cv2`)
* `open3d`
* `torch`
* `scipy`

## Usage

1. **Setup Environment**: Install dependencies and the DA3 library as linked above.
2. **Prepare Data**: Place your images or video in a data folder (e.g., `./data/room_photos`).
3. **Run Notebook**: Open `main.ipynb` and execute the cells in order.
* The `setup_paths` function will automatically create necessary `results` and `masks` directories.
* The final steps will export your 3D models to the results directory.
