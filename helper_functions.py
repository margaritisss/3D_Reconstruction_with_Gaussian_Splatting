import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

def setup_paths(data_folder = "your_data_folder"):
    
    paths = {
        'data':    f"./data/{data_folder}",
        'results': f"./results/{data_folder}",
        'masks':   f"./data/{data_folder}/masks",
    }
    os.makedirs(paths['results'], exist_ok = True)
    os.makedirs(paths['masks'],   exist_ok = True)
    
    return paths


def visualize_depth_and_confidence(images, depths, confidences, sample_idx = 0):
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(images[sample_idx]) 
    axs[0].set_title('RGB Image')
    axs[0].axis('off')
    
    axs[1].imshow(depths[sample_idx], cmap='turbo')
    axs[1].set_title('Depth Map')
    axs[1].axis('off')
    
    axs[2].imshow(confidences[sample_idx], cmap='viridis')
    axs[2].set_title('Confidence Map')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def visualize_point_cloud_open3d(points, colors = None, window_name = "Point Cloud"):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name = window_name)


def save_point_cloud(points, colors=None, save_path="output.ply"):
    
    # 1. Create the object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 2. SAVE the file (This is the new part)
    # By providing just a filename (e.g. "my_scan.ply"), it saves in the current folder.
    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Saved point cloud to: {save_path}")