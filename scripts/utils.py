# SPDX-FileCopyrightText: 2025 Mobile Robotics Lab, Technical University of Munich
# SPDX-FileCopyrightText: 2025 Jiaxin Wei
# SPDX-License-Identifier: Apache-2.0


import os
import cv2
import json
import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import RotationSpline
import torchvision.transforms.functional as F
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def collect_files(folder_path):
    files = sorted(os.listdir(folder_path))
    results = [os.path.join(folder_path, files[i]) for i in range(len(files))]
    return results


def eval_image(gt_images, gs_path, results_path):
    gs_results = collect_files(gs_path)

    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    gs_psnr_scores = []
    gs_ssim_scores = []
    gs_lpips_scores = []

    for i in range(len(gt_images)):
        gt_image = cv2.imread(gt_images[i]) / 255
        gt_image = F.to_tensor(gt_image).to(torch.float32).unsqueeze(0) 

        rendered_gs_image = cv2.imread(gs_results[i]) / 255
        rendered_gs_image = F.to_tensor(rendered_gs_image).to(torch.float32).unsqueeze(0)

        psnr_score = psnr(rendered_gs_image, gt_image)
        gs_psnr_scores.append(psnr_score)
        ssim_score = ssim(rendered_gs_image, gt_image)
        gs_ssim_scores.append(ssim_score)
        lpips_score = lpips(rendered_gs_image, gt_image)
        gs_lpips_scores.append(lpips_score)

    mean_gs_scores = {
        "psnr": float(torch.stack(gs_psnr_scores).mean().item()),
        "ssim": float(torch.stack(gs_ssim_scores).mean().item()),
        "lpips": float(torch.stack(gs_lpips_scores).mean().item()),
    }

    for key, value in mean_gs_scores.items():
        print(f"{key}: {value}")

    with open(results_path, "w") as file:
        print(f"Saving results to {results_path}")
        json.dump(mean_gs_scores, file, indent=2)


def render_mesh(mesh, intrinsics, R_WC, t_WC):
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    width = intrinsics["width"]
    height = intrinsics["height"]

    # Generate ray directions for each pixel
    xv, yv = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    x = (xv - cx) / fx
    y = (yv - cy) / fy
    z = np.ones_like(x)

    camera_rays = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    camera_rays = camera_rays.reshape(-1, 3)  # Flatten to (N, 3)
    camera_rays /= np.linalg.norm(camera_rays, axis=1, keepdims=True)  # Normalize
    N = camera_rays.shape[0]

    # Transform rays to world space
    ray_directions = (R_WC @ camera_rays.T).T  # Rotate rays to world frame
    ray_directions /= np.linalg.norm(ray_directions, axis=1, keepdims=True)  # Normalize

    # Create Open3D ray tensor
    rays = np.hstack([np.tile(t_WC, (ray_directions.shape[0], 1)), ray_directions])
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    # Perform raycasting
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    ans = scene.cast_rays(rays)
    
    t_hit = ans['t_hit'].numpy()  # (N,)
    t_hit = t_hit * camera_rays[:, 2]  # turn ray length into z-depth
    primitive_ids = ans['primitive_ids'].numpy()  # (N,)
    primitive_uvs = ans['primitive_uvs'].numpy()  # (N, 2); (u,v) barycentrics.

    # Prepare output buffers.
    color = np.zeros((N, 3), dtype=np.float32)
    triangles = np.asarray(mesh.triangles)  # shape: (F, 3)
    vertex_colors = np.asarray(mesh.vertex_colors)  # shape: (V, 3) (assumed normalized to [0,1])

    valid = t_hit < np.inf  # valid hit if t_hit is finite.
    valid_idx = np.where(valid)[0]
    # For each valid ray, get the face index.
    face_ids = primitive_ids[valid_idx].astype(np.int32)
    # Get vertex indices for the intersected faces.
    face_vertices = triangles[face_ids]  # (M, 3)
    # Compute barycentrics: [1 - u - v, u, v].
    uv = primitive_uvs[valid_idx]  # (M, 2)
    bary = np.hstack([1 - uv[:, 0:1] - uv[:, 1:2], uv])  # (M, 3)

    # Interpolate color.
    color0 = vertex_colors[face_vertices[:, 0]]
    color1 = vertex_colors[face_vertices[:, 1]]
    color2 = vertex_colors[face_vertices[:, 2]]
    interp_color = bary[:, 0:1] * color0 + bary[:, 1:2] * color1 + bary[:, 2:3] * color2
    color[valid_idx] = interp_color
    color = color.reshape(height, width, 3)
    color = np.clip(color, 0, 1) * 255

    return color


def read_replica_cameras(path, intrinsics_only=False):
    parent_path = os.path.dirname(path)
    intrinsics_path = os.path.join(parent_path, "cam_params.json")
    with open(intrinsics_path, 'r') as file:
        data = json.load(file)

    intrinsics = {}
    intrinsics["width"] = data["camera"]["w"]
    intrinsics["height"] = data["camera"]["h"]
    intrinsics["fx"] = data["camera"]["fx"]
    intrinsics["fy"] = data["camera"]["fy"]
    intrinsics["cx"] = data["camera"]["cx"]
    intrinsics["cy"] = data["camera"]["cy"]

    if intrinsics_only:
        return intrinsics

    gt_path = os.path.join(path, "traj.txt")
    with open(gt_path, 'r') as file: 
        gt_infos = file.readlines()

    novel_view_path = os.path.join(path, "novel_views.json")
    if not os.path.exists(novel_view_path):
        raise FileNotFoundError(f"{novel_view_path} doesn't exist!")
    with open(novel_view_path, "r") as f:
        novel_views = json.load(f)

    train_cameras = []
    for gt_info in gt_infos:
        gt_info = gt_info.strip().split()
        R_wc = np.array([[float(gt_info[0]), float(gt_info[1]), float(gt_info[2])],
                         [float(gt_info[4]), float(gt_info[5]), float(gt_info[6])],
                         [float(gt_info[8]), float(gt_info[9]), float(gt_info[10])]])
        R_wc = R.from_matrix(R_wc).as_matrix()
        t_wc = np.array([float(gt_info[3]), float(gt_info[7]), float(gt_info[11])])
        T_wc = np.eye(4)
        T_wc[:3, :3] = R_wc
        T_wc[:3, 3] = t_wc
        train_cameras.append(T_wc)

    test_cameras = []
    for i in range(len(novel_views)):
        T_wc = np.linalg.inv(np.array(novel_views[i]["extrinsic"]))
        test_cameras.append(T_wc)
    
    train_cameras = np.stack(train_cameras)
    test_cameras = np.stack(test_cameras)

    return train_cameras, test_cameras, intrinsics
        

def read_scannetpp_cameras(path, intrinsics_only=False):
    with open(os.path.join(path, "nerfstudio/transforms_undistorted_2.json"), 'r') as gt_f:
        gt = json.load(gt_f)
    
    intrinsics = {}
    intrinsics["fx"] = gt["fl_x"]
    intrinsics["fy"] = gt["fl_y"]
    intrinsics["cx"] = gt["cx"]
    intrinsics["cy"] = gt["cy"]
    intrinsics["width"] = gt['w']
    intrinsics["height"] = gt['h']

    if intrinsics_only:
        return intrinsics

    train_frames = sorted(gt["frames"], key = lambda x : x["file_path"])
    test_frames = sorted(gt["test_frames"], key = lambda x : x["file_path"])

    train_cameras = []
    for i in range(len(train_frames)):
        T_wc = np.array(train_frames[i]["transform_matrix"])
        P = np.eye(4)
        P[1, 1] = -1
        P[2, 2] = -1
        T_wc = np.dot(T_wc, P)
        train_cameras.append(T_wc)

    test_cameras = []
    for i in range(len(test_frames)):
        T_wc = np.array(test_frames[i]["transform_matrix"])
        P = np.eye(4)
        P[1, 1] = -1
        P[2, 2] = -1
        T_wc = np.dot(T_wc, P)
        test_cameras.append(T_wc)
    
    train_cameras = np.stack(train_cameras)
    test_cameras = np.stack(test_cameras)
    
    return train_cameras, test_cameras, intrinsics
