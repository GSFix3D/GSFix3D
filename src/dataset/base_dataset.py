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


import cv2
import os
import copy
import random
import torch
import numpy as np
from PIL import Image
from enum import Enum
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize


class DatasetMode(Enum):
    TRAIN = "train"
    EVAL = "evaluate"
    VIS = "vis"


class BaseDataset(Dataset):
    """
    Base dataset providing shared utilities and common fields used by subclasses.

    Features:
      - Initialization arguments: mode (DatasetMode), dataset_dir, disp_name,
        dual_input, resize_to_hw, rgb_transform, and optional filename_ls_path.
      - Loads a semantic-mask dataset ("paint-by-inpaint/PIPE_Masks") for augmentations.
      - Helper methods:
          _read_image(path) -> np.ndarray: open image and return HxWxC numpy array.
          _prepare_image(image) -> (torch.IntTensor, torch.FloatTensor): convert uint8
            image to an int tensor and a normalized float tensor in [-1, 1].
          _training_preprocess(rasters): resize rasters when resize_to_hw is set.
          _sample_mask(split, width, height): sample a sparse semantic mask and a
            blurred version resized to (width, height) (ensures foreground < 10%).
      - Subclasses (BaseTrainDataset, BaseFinetuneDataset) implement dataset-specific
        loading, indexing and augmentation logic.
    """

    def __init__(
        self,
        mode: DatasetMode,
        dataset_dir: str,
        disp_name: str,
        dual_input: bool = False,
        resize_to_hw=None,
        rgb_transform=lambda x: x / 255.0 * 2 - 1,
        filename_ls_path: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.dataset_dir = dataset_dir
        assert os.path.exists(self.dataset_dir), f"Dataset does not exist at: {self.dataset_dir}"
        self.disp_name = disp_name
        self.dual_input = dual_input

        self.resize_to_hw = resize_to_hw
        self.rgb_transform = rgb_transform

        # optional filenames list (used by BaseTrainDataset)
        self.filename_ls_path = filename_ls_path
        if self.filename_ls_path is not None:
            with open(self.filename_ls_path, "r") as f:
                self.filenames = [s.split() for s in f.readlines()]

        # Load semantic masks for augmentation
        data_files = {"train": "data/train-*", "test": "data/test-*"}
        self.dataset_masks = load_dataset("paint-by-inpaint/PIPE_Masks", data_files=data_files)

    def _read_image(self, image_path) -> np.ndarray:
        """
        Open image from filesystem.
        Returns numpy array HxWxC.
        """
        image = Image.open(image_path)
        return np.asarray(image)
    
    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image to torch tensor.
        """
        image = np.transpose(image, (2, 0, 1)).astype(int)  # [C, H, W]
        image_norm = image / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
        image = torch.from_numpy(image).int()
        image_norm = torch.from_numpy(image_norm).float()
        return image, image_norm   
    
    def _training_preprocess(self, rasters):
        """
        Shared preprocessing.
        """
        # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.BILINEAR
            )
            rasters = {k: resize_transform(v) for k, v in rasters.items()}
        return rasters
    
    def _sample_mask(self, split, width, height):
        while True:
            idx = random.randint(0, len(self.dataset_masks[split]) - 1)
            m = self.dataset_masks[split][idx]["mask"]
            m = np.asarray(m.convert("L"))
            ratio = np.sum(m > 0) / (m.shape[0] * m.shape[1])
            if ratio < 0.1:
                m = np.repeat(m[..., np.newaxis], 3, axis=-1) / 255.0
                m = cv2.resize(m, (width, height))
                blurred_m = cv2.GaussianBlur(m, (51, 51), sigmaX=10, sigmaY=10)
                return m, blurred_m


class BaseTrainDataset(BaseDataset):
    def __init__(
        self,
        mode: DatasetMode,
        dataset_dir: str,
        disp_name: str,
        dual_input: bool = False,
        resize_to_hw=None,
        rgb_transform=lambda x: x / 255.0 * 2 - 1,  #  [0, 255] -> [-1, 1],
        filename_ls_path: str = None,
        **kwargs,
    ) -> None:
        
        if filename_ls_path is None:
            raise ValueError("filename_ls_path must be provided for BaseTrainDataset.")
    
        super().__init__(
            mode=mode,
            filename_ls_path=filename_ls_path,
            dataset_dir=dataset_dir,
            disp_name=disp_name,
            dual_input=dual_input,
            resize_to_hw=resize_to_hw,
            rgb_transform=rgb_transform,
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        rgb_rel_path = self._get_data_path(index=index)
        rasters = {}
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))
        other = {"index": index, "rgb_path": os.path.join(self.dataset_dir, rgb_rel_path)}
        return rasters, other

    def _get_data_path(self, index):
        filename_line = self.filenames[index]
        rgb_rel_path = filename_line[0]
        return rgb_rel_path

    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_image(os.path.join(self.dataset_dir, rgb_rel_path))

        # Random semantic mask augmentation
        split = "train" if DatasetMode.TRAIN == self.mode else "test"
        if random.random() < 0.3:
            mask, blurred_mask = self._sample_mask(split, rgb.shape[1], rgb.shape[0])
            gs_image = rgb * (1 - blurred_mask)
            if self.dual_input:
                mesh_image = rgb * (1 - mask)
        else:
            gs_image = copy.deepcopy(rgb)
            if self.dual_input:
                mesh_image = copy.deepcopy(rgb)

        # Further random mask for gs_image
        if random.random() < 0.3:
            _, blurred_mask = self._sample_mask(split, rgb.shape[1], rgb.shape[0])
            gs_image = gs_image * (1 - blurred_mask)

        rgb, rgb_norm = self._prepare_image(rgb)
        gs_image, gs_norm = self._prepare_image(gs_image)
        outputs = {
            "rgb_int": rgb,
            "rgb_norm": rgb_norm,
            "gs_int": gs_image,
            "gs_norm": gs_norm,
        }
        
        if self.dual_input:
            mesh_image, mesh_norm = self._prepare_image(mesh_image)
            outputs.update({"mesh_int": mesh_image,
                            "mesh_norm": mesh_norm})

        return outputs


class BaseFinetuneDataset(BaseDataset):
    def __init__(
        self,
        mode: DatasetMode,
        dataset_dir: str,
        disp_name: str,
        dual_input: bool = False,
        resize_to_hw=None,
        rgb_transform=lambda x: x / 255.0 * 2 - 1,  #  [0, 255] -> [-1, 1],
        **kwargs,
    ) -> None:
        super().__init__(
            mode=mode,
            dataset_dir=dataset_dir,
            disp_name=disp_name,
            dual_input=dual_input,
            resize_to_hw=resize_to_hw,
            rgb_transform=rgb_transform,
        )

        # Load dataset file paths for finetuning
        def _list_images_in_dir(d):
            allowed_exts = {".jpg", ".jpeg", ".png"}
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Directory not found: {d}")
            files = [os.path.join(d, f) for f in os.listdir(d)
                if os.path.isfile(os.path.join(d, f)) and os.path.splitext(f)[1].lower() in allowed_exts
            ]
            return sorted(files)

        self.gt_rgb_path = _list_images_in_dir(os.path.join(self.dataset_dir, "rgb"))
        self.rendered_gs_path = _list_images_in_dir(os.path.join(self.dataset_dir, "rendered_gs"))

        assert len(self.gt_rgb_path) == len(self.rendered_gs_path), "The number of images doesn't match!"

        if self.dual_input:
            self.rendered_mesh_path = _list_images_in_dir(os.path.join(self.dataset_dir, "rendered_mesh"))
            assert len(self.gt_rgb_path) == len(self.rendered_mesh_path), "The number of images doesn't match!"
        
        n_samples = len(self.gt_rgb_path)
        if self.mode == DatasetMode.VIS:
            vis_num = int(0.01 * n_samples)
            self.gt_rgb_path = self.gt_rgb_path[:vis_num]
            self.rendered_gs_path = self.rendered_gs_path[:vis_num]
            if self.dual_input:
                self.rendered_mesh_path = self.rendered_mesh_path[:vis_num]
            
        elif self.mode == DatasetMode.EVAL:
            eval_num = int(0.1 * n_samples)
            self.gt_rgb_path = self.gt_rgb_path[:eval_num]
            self.rendered_gs_path = self.rendered_gs_path[:eval_num]
            if self.dual_input:
                self.rendered_mesh_path = self.rendered_mesh_path[:eval_num]

        self.n_samples = len(self.gt_rgb_path)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        gt_rgb_file = self.gt_rgb_path[index]   
        rendered_gs_file = self.rendered_gs_path[index]

        gt_rgb = self._read_image(gt_rgb_file)
        rendered_gs = self._read_image(rendered_gs_file)
        
        if self.dual_input:
            rendered_mesh_file = self.rendered_mesh_path[index]
            rendered_mesh = self._read_image(rendered_mesh_file)
        
        # Random semantic mask augmentation
        if random.random() < 0.3:
            mask, blurred_mask = self._sample_mask("test", gt_rgb.shape[1], gt_rgb.shape[0])
            rendered_gs = rendered_gs * (1 - blurred_mask)
            if self.dual_input:
                rendered_mesh = rendered_mesh * (1 - mask)

        # Further random mask for gs_image
        if random.random() < 0.3:
            _, blurred_mask = self._sample_mask("test", gt_rgb.shape[1], gt_rgb.shape[0])
            rendered_gs = rendered_gs * (1 - blurred_mask)

        gt_rgb, gt_rgb_norm = self._prepare_image(gt_rgb)
        rendered_gs, rendered_gs_norm = self._prepare_image(rendered_gs)  
        outputs = {
            "gt_rgb": gt_rgb,
            "gt_rgb_norm": gt_rgb_norm,
            "rendered_gs": rendered_gs,
            "rendered_gs_norm": rendered_gs_norm,
        }

        if self.dual_input:
            rendered_mesh, rendered_mesh_norm = self._prepare_image(rendered_mesh)
            outputs.update({
                "rendered_mesh": rendered_mesh,
                "rendered_mesh_norm": rendered_mesh_norm,
            })  

        if DatasetMode.TRAIN == self.mode:
            outputs = self._training_preprocess(outputs)
            
        outputs.update({"index": index, "rgb_path": gt_rgb_file})
        
        return outputs
