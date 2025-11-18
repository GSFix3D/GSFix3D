# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------
# This file has been modified by Mobile Robotics Lab, TU Munich (2025).
# SPDX-FileCopyrightText: 2025 Jiaxin Wei
# SPDX-License-Identifier: Apache-2.0


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import argparse
import logging
import json
import torch
from PIL import Image
from glob import glob
from tqdm import tqdm

from marigold import MarigoldGSFixerPipeline, MarigoldGSFixerOutput
from scripts.utils import collect_files, eval_image


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="GSFixer: Multi-image Inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="goldoak1421/gsfixer-full-replica-room1",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--recon_method_type",
        type=str,
        default="gsfusion",
        help="File name postfix to locate input gs images. We also provide data from 'splatam' and 'rtg'."
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="replica",
        choices=["replica", "scannetpp"]
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_gsfixer",
        help="Output directory."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Enable evaluation."
    )
    parser.add_argument(
        "--dual_input",
        action="store_true",
        help="Enable conditioning on mesh input."
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. If set to "
        "`None`, default value will be read from checkpoint.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Resolution to which the input is resized before performing estimation. `0` uses the original input "
        "resolution; `None` resolves the best default from the model checkpoint. Default: `None`",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="Setting this flag will output the result at the effective value of `processing_res`, otherwise the "
        "output will be resized to the input resolution.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or "
        "`nearest`. Default: `bilinear`",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for randomized inference. Default: `None`",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Use Apple Silicon for faster inference (subject to availability).",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, "
            "due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    if args.eval:
        if args.data_type == "replica":
            gt_rgb_file_paths = sorted(glob(os.path.join(args.data_path, "novel_views", "frame*.jpg")))
        elif args.data_type == "scannetpp":
            split_file = os.path.join(args.data_path, "train_test_lists.json")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"{split_file} doesn't exist!")
            with open(split_file, "r") as f:
                splits = json.load(f)
            gt_rgb_file_paths = sorted([os.path.join(args.data_path, "undistorted_images_2", file_name) for file_name in splits["test"]])
        else:
            raise TypeError(f"Unsupported dataset type: {args.data_type}!")

    # Collect all files need to be fixed
    input_gs_paths = os.path.join(args.data_path, "rendered_novel_views", "gs_image_" + args.recon_method_type)
    rendered_gs_files = collect_files(input_gs_paths)
    if args.dual_input:
        rendered_mesh_files = collect_files(os.path.join(args.data_path, "rendered_novel_views", "mesh_image"))
    logging.info(f"Found {len(rendered_gs_files)} images for fixing")

    # -------------------- Model --------------------
    pipe: MarigoldGSFixerPipeline = MarigoldGSFixerPipeline.from_pretrained(checkpoint_path)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

    pipe = pipe.to(device)
    logging.info(f"Loaded GSFixer pipeline.")

    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}."
    )

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for i, gs_image_path in tqdm(enumerate(rendered_gs_files), total=len(rendered_gs_files)):
            # Read input image
            input_gs_image = Image.open(gs_image_path)

            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # Perform inference
            if args.dual_input:
                input_mesh_image = Image.open(rendered_mesh_files[i])

                pipe_out: MarigoldGSFixerOutput = pipe(
                input_gs_image,
                input_mesh_image,
                denoising_steps=denoise_steps,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                show_progress_bar=False,
                resample_method=resample_method,
                generator=generator,
            )
            else:
                pipe_out: MarigoldGSFixerOutput = pipe(
                    input_gs_image,
                    denoising_steps=denoise_steps,
                    processing_res=processing_res,
                    match_input_res=match_input_res,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    resample_method=resample_method,
                    generator=generator,
                )

            fixed_rgb: Image.Image = pipe_out.fixed_rgb

            # Save as image
            fixed_rgb.save(os.path.join(output_dir, "rgb", os.path.basename(gs_image_path)))

    if args.eval:
        print("-----Evaluation results before fixing-----")
        eval_image(gt_rgb_file_paths, input_gs_paths, os.path.join(args.output_dir, "before_metrics.json"))

        print("-----Evaluation results after fixing-----")
        eval_image(gt_rgb_file_paths, os.path.join(output_dir, "rgb"), os.path.join(args.output_dir, "after_metrics.json"))