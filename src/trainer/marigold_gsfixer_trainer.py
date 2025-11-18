# SPDX-FileCopyrightText: 2025 Mobile Robotics Lab, Technical University of Munich
# SPDX-FileCopyrightText: 2025 Jiaxin Wei
# SPDX-License-Identifier: Apache-2.0


from typing import List
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from marigold.marigold_gsfixer_pipeline import MarigoldGSFixerPipeline
from src.trainer.base_trainer import BaseTrainer


class MarigoldGSFixerTrainer(BaseTrainer):
    def __init__(
        self,
        cfg: OmegaConf,
        model: MarigoldGSFixerPipeline,
        train_dataloader: DataLoader,
        device,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
        dual_input: bool = False,
    ):
        super().__init__(cfg,
                         model,
                         train_dataloader,
                         device,
                         out_dir_ckpt,
                         out_dir_eval,
                         out_dir_vis,
                         accumulation_steps,
                         val_dataloaders,
                         vis_dataloaders,
                         dual_input,
                        )

    def _load_train_data(self, batch):
        """Load data from batch for training from scratch."""
        rgb = batch["rgb_norm"].to(self.device)
        corrupted_rgb1 = batch["gs_norm"].to(self.device)
        corrupted_rgb1 = self.corrupt_transforms(corrupted_rgb1)
        corrupted_rgb1 += torch.randn_like(corrupted_rgb1) * 0.2
        corrupted_rgb1 = torch.clamp(corrupted_rgb1, -1.0, 1.0)
        if self.dual_input:   
            corrupted_rgb2 = batch["mesh_norm"].to(self.device)
            corrupted_rgb2 = self.corrupt_transforms(corrupted_rgb2)
            corrupted_rgb2 += torch.randn_like(corrupted_rgb2) * 0.2
            corrupted_rgb2 = torch.clamp(corrupted_rgb2, -1.0, 1.0)
        else:
            corrupted_rgb2 = None

        return rgb, corrupted_rgb1, corrupted_rgb2
    
    def _load_val_data(self, batch):
        """Load validation/visualization data from batch."""
        rgb = batch["rgb_int"] / 255.0
        corrupted_rgb1 = batch["gs_int"] / 255.0
        corrupted_rgb1 = self.corrupt_transforms(corrupted_rgb1)
        corrupted_rgb1 += torch.randn_like(corrupted_rgb1) * 0.2
        corrupted_rgb1 = torch.clamp(corrupted_rgb1, 0.0, 1.0) * 255.0
        if self.dual_input:   
            corrupted_rgb2 = batch["mesh_int"] / 255.0
            corrupted_rgb2 = self.corrupt_transforms(corrupted_rgb2)
            corrupted_rgb2 += torch.randn_like(corrupted_rgb2) * 0.2
            corrupted_rgb2 = torch.clamp(corrupted_rgb2, 0.0, 1.0) * 255.0
        else:
            corrupted_rgb2 = None

        return rgb, corrupted_rgb1, corrupted_rgb2
