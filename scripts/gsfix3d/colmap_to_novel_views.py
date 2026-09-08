# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: Apache-2.0
#
# Replaces the interactive novel_view_capture.py for COLMAP data.
# Reads existing COLMAP camera poses, exports novel_views.json,
# and renders GS images from each selected pose.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from gs.camera import Camera
from gs.gaussian_model import GaussianModel
from gs.gaussian_renderer import render
from gs.general_utils import searchForMaxIteration, focal2fov
from gs.arguments import ModelParams, PipelineParams, get_combined_args
from scripts.utils import read_colmap_cameras


def main(args, model_params, pipeline_params):
    # Load camera poses from COLMAP
    train_cameras, _, intrinsics = read_colmap_cameras(args.data_path)
    print(f"Loaded {len(train_cameras)} cameras from COLMAP.")

    # Select a subset using stride (every Nth camera)
    indices = list(range(0, len(train_cameras), args.stride))
    if args.max_views > 0:
        indices = indices[:args.max_views]
    print(f"Selected {len(indices)} cameras (stride={args.stride}).")

    # Build novel_views.json (T_CW extrinsic matrices)
    novel_views = []
    for idx in indices:
        T_WC = train_cameras[idx]
        T_CW = np.linalg.inv(T_WC)
        novel_views.append({"extrinsic": T_CW.tolist()})

    poses_file = os.path.join(args.output_dir, "novel_views.json")
    with open(poses_file, "w") as f:
        json.dump(novel_views, f, indent=4)
    print(f"Saved {len(novel_views)} poses to {poses_file}")

    # Set up renderer
    if args.iteration == -1:
        loaded_iter = searchForMaxIteration(os.path.join(model_params.model_path, "point_cloud"))
    else:
        loaded_iter = args.iteration

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        raise Exception("No GPU found!")

    gaussians = GaussianModel(model_params.sh_degree)
    gaussians.load_ply(os.path.join(model_params.model_path,
                                    "point_cloud",
                                    "iteration_" + str(loaded_iter),
                                    "point_cloud.ply"))

    fov_x = focal2fov(intrinsics["fx"], intrinsics["width"])
    fov_y = focal2fov(intrinsics["fy"], intrinsics["height"])
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    gs_image_dir = os.path.join(args.output_dir, "rendered_novel_views", "gs_image")
    os.makedirs(gs_image_dir, exist_ok=True)

    print("Rendering GS images from selected poses...")
    for i, pose in tqdm(enumerate(novel_views), total=len(novel_views)):
        T_CW = np.array(pose["extrinsic"])
        gs_cam = Camera(
            R=T_CW[:3, :3], T=T_CW[:3, 3],
            FoVx=fov_x, FoVy=fov_y,
            width=intrinsics["width"], height=intrinsics["height"]
        )
        with torch.no_grad():
            rendered = render(gs_cam, gaussians, pipeline_params, background)["render"]
            img = rendered.detach().cpu().numpy().transpose(1, 2, 0) * 255
            img = img.astype(np.uint8)
            Image.fromarray(img).save(os.path.join(gs_image_dir, f"{i:05d}.png"))

    print(f"Done. Rendered images saved to {gs_image_dir}")
    print(f"Novel views JSON saved to {poses_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COLMAP cameras to novel views for GSFix3D")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--data_path", type=str, required=True, help="Path to folder with cameras.txt and images.txt")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder for novel_views.json and rendered images")
    parser.add_argument("--stride", type=int, default=10, help="Use every Nth camera (default: 10)")
    parser.add_argument("--max_views", type=int, default=0, help="Max number of views to use (0 = no limit)")
    parser.add_argument("--iteration", type=int, default=-1, help="Which 3DGS iteration to load (-1 = latest)")

    args = get_combined_args(parser)
    os.makedirs(args.output_dir, exist_ok=True)

    main(args, model.extract(args), pipeline.extract(args))
