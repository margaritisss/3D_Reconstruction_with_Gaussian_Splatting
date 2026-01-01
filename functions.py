import torch
import cv2
import os
import glob
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from depth_anything_3.api import DepthAnything3


def load_da3_model(model_name = "depth-anything/DA3NESTED-GIANT-LARGE"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device)
    
    return model, device


def load_images_from_folder(data_folder):
    
    """ scan folder and load all images """
    
    images = [] # list of images
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    for ext in extensions:
        images.extend(sorted(glob.glob(os.path.join(data_folder, ext))))
    print(f"Found {len(images)} images in {data_folder}")
    return images


def extract_frames_from_video(video_path, output_folder, frame_rate = 1):
    
    """ extract frames from video at specified frame rate """
    
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_rate == 0:
            output_path = os.path.join(output_folder, f"frame_{saved_count:05d}.png")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
        
    cap.release()
    print(f"Extracted {saved_count} frames to {output_folder}")
    return output_folder      
 

def run_da3_model(model, images, process_res_method = "upper_bound_resize"):
    
    """ Run Depth Anything 3 model to get depth maps, camera intrinsics and poses """
    
    predict = model.inference(
        image              = images,
        infer_gs           = True,
        process_res_method = process_res_method
    )
    
    print(f" Depth maps shape: {predict.depth.shape}")
    print(f" Extrinsics shape: {predict.extrinsics.shape}")
    print(f" Intrinsics shape: {predict.intrinsics.shape}")
    print(f" Confidence shape: {predict.conf.shape}")
    
    return predict


def depth_2_pointcloud(depth_map, rgb_image, intrinsics, extrinsics, conf_map = None, conf_threshold = 0.8):
    
    """ Projecting depth map to 3d points using camera intrinsics and extrinsics """
    
    h, w = depth_map.shape
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    
    # create meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # filtering by confidence
    if conf_map is not None:
        
        valid_mask = conf_map > conf_threshold
        x          = x[valid_mask]
        y          = y[valid_mask]
        depth_map  = depth_map[valid_mask]   
        rgb_image  = rgb_image[valid_mask]
        
    else:
        
        x = x.flatten()
        y = y.flatten()
        depth_map = depth_map.flatten()
        rgb_image = rgb_image.reshape(-1, 3) 
    
    # back-project to 3D
    u = (x - cx) * depth_map / fx
    v = (y - cy) * depth_map / fy
    z = depth_map
    
    points_3d = np.stack([u, v, z], axis=-1) # np.stack stacks along a new axis, axis=-1 means last axis
    
    # transform points to world coordinates
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    world_points = (points_3d - t) @ R
    
    colours = rgb_image.astype(np.float32) / 255.0
    
    return world_points, colours
    
    
def merge_pointclouds(prediction, conf_threshold = 0.8):
    
    """ Merge point clouds from all depth maps """
    
    all_points  = []
    all_colours = []
    
    n_frames = len(prediction.depth)
    
    for i in range(n_frames):
        
        points, colours = depth_2_pointcloud(
            prediction.depth[i],
            prediction.processed_images[i],
            prediction.intrinsics[i],
            prediction.extrinsics[i],
            prediction.conf[i],
            conf_threshold         )
        
        all_points.append(points)
        all_colours.append(colours)
        
    merged_points  = np.vstack(all_points)
    merged_colours = np.vstack(all_colours)
    
    print(f"Merged point cloud has {len(merged_points)} points.")   
    return merged_points, merged_colours   


def clean_point_cloud(points_3d, colours_3d, nb_neighbors=20, std_ratio=2.0):
    
    """ Clean point cloud using statistical outlier removal """
    
    # 1. Converting numpy arrays to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # open3d expects colors in range [0,1]. If colours_3d are in [0,255], we need to normalize them.
    if colours_3d.max() > 1.0:
        pcd.colors = o3d.utility.Vector3dVector(colours_3d / 255.0)
    else:
        pcd.colors = o3d.utility.Vector3dVector(colours_3d)
    
    # 2. Statistical outlier removal
    # cl  = the cleaned point cloud
    # ind = the indices of the points that are kept
    
    cl , ind = pcd.remove_statistical_outlier(nb_neighbors = nb_neighbors,
                                              std_ratio    = std_ratio)
    
    # 3. we use the indices to filter the original colours_3d array
    # this ensures that the colors correspond to the cleaned points
    
    inlier_mask     = np.asarray(ind)
    cleaned_points  = points_3d[inlier_mask]
    cleaned_colours = colours_3d[inlier_mask]
    
    return cleaned_points, cleaned_colours


def clean_point_cloud_scipy(points_3d, colours_3d, nb_neighbors=20, std_ratio=2.0):
    
    # 1. build a KDTree for fast neighbor search
    tree = cKDTree(points_3d)
    
    # 2. query neighbors
    # k needs to be nb_neighbors + 1 because the point itself is included in results
    distances, _ = tree.query(points_3d, k = nb_neighbors + 1, workers = -1)
    
    # exclude the first column which is distance to itself (zero)
    mean_distances = np.mean(distances[:, 1:], axis=1)  
    
    # 3. calculating statistics
    global_mean = np.mean(mean_distances) 
    global_std  = np.std(mean_distances)
    
    # 4. generating the mask
    distance_threshold = global_mean + (std_ratio * global_std)
    mask = mean_distances < distance_threshold
    
    return points_3d[mask], colours_3d[mask]


def export_da3_outputs(model, images, output_dir, conf_thresh_percentile = 40.0):
    """Export GLB with cameras and Gaussian Splatting PLY using DA3 built-in export"""
    prediction = model.inference(
        image = images,
        infer_gs=True,
        process_res_method="upper_bound_resize",
        export_dir=output_dir,
        export_format="glb-gs_ply",
        conf_thresh_percentile=conf_thresh_percentile,
        num_max_points=10_000_000,
        show_cameras=True
    )
    
    print(f"Exported to: {output_dir}")
    return prediction