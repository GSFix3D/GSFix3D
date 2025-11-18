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


from .base_dataset import BaseTrainDataset


class VirtualKITTIDataset(BaseTrainDataset):
    def __init__(
        self,
        kitti_bm_crop,  # Crop to KITTI benchmark size
        valid_mask_crop,  # Evaluation mask. [None, garg or eigen]
        **kwargs,
    ) -> None:
        super().__init__(
           **kwargs,
        )
        self.kitti_bm_crop = kitti_bm_crop
        self.valid_mask_crop = valid_mask_crop
        assert self.valid_mask_crop in [
            None,
            "garg",  # set evaluation mask according to Garg  ECCV16
            "eigen",  # set evaluation mask according to Eigen NIPS14
        ], f"Unknown crop type: {self.valid_mask_crop}"

        # Filter out empty depth
        self.filenames = [f for f in self.filenames if "None" != f[1]]

    def _load_rgb_data(self, rgb_rel_path):
        rgb_data = super()._load_rgb_data(rgb_rel_path)
        if self.kitti_bm_crop:
            rgb_data = {
                k: VirtualKITTIDataset.kitti_benchmark_crop(v)
                for k, v in rgb_data.items()
            }
        return rgb_data
    
    @staticmethod
    def kitti_benchmark_crop(input_img):
        """
        Crop images to KITTI benchmark size
        Args:
            `input_img` (torch.Tensor): Input image to be cropped.

        Returns:
            torch.Tensor:Cropped image.
        """
        KB_CROP_HEIGHT = 352
        KB_CROP_WIDTH = 1216

        height, width = input_img.shape[-2:]
        top_margin = int(height - KB_CROP_HEIGHT)
        left_margin = int((width - KB_CROP_WIDTH) / 2)
        if 2 == len(input_img.shape):
            out = input_img[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        elif 3 == len(input_img.shape):
            out = input_img[
                :,
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        return out
