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


import logging
import numpy as np
from typing import Optional, Union
import torch
from PIL import Image
from tqdm.auto import tqdm
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from transformers import CLIPTextModel, CLIPTokenizer

from .util.batchsize import find_batch_size
from .util.image_util import (
    chw2hwc,
    get_tv_resample_method,
    resize_max_res,
)


class MarigoldGSFixerOutput(BaseOutput):
    """
    Output class for the MarigoldGSFixerPipeline.

    Args:
        fixed_rgb (None or PIL.Image.Image):
            Final fixed RGB image as a PIL Image with uint8 values in [0, 255].
            This can be None if no image is produced.
        fixed_rgb_ts (torch.Tensor):
            Tensor representation of the fixed RGB image with float values in [0.0, 1.0].
    """

    fixed_rgb: Union[None, Image.Image]
    fixed_rgb_ts: torch.Tensor


class MarigoldGSFixerPipeline(DiffusionPipeline):
    """
    Pipeline for Marigold GSFixer: a conditional diffusion pipeline that takes one or two RGB images as conditions
    and generates a fixed RGB output image.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Behavior summary:
      - Encodes input RGB image(s) with the provided VAE into latents.
      - Runs a denoising diffusion process in latent space with the provided UNet and scheduler.
      - Decodes the final latent to an RGB image via the VAE decoder and returns both a PIL image and a float-tensor
        in range [0.0, 1.0].

    Important implementation details:
      - If two condition images are provided, the UNet input is concatenated as
        [condition_rgb2_latent(mesh), condition_rgb1_latent(gs), target_latent].
        If only one condition image is provided, the UNet input is concatenated as
        [condition_rgb1_latent(gs), target_latent].
      - The pipeline uses an empty text embedding (encoded via the provided CLIP text_encoder/tokenizer) as
        encoder_hidden_states for the conditional UNet call.
      - Latent scaling is handled by `latent_scale_factor` when encoding/decoding through the VAE.

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net that predicts noise residuals in latent space.
        vae (`AutoencoderKL`):
            VAE used to encode input images to latents and decode latents back to RGB.
        scheduler (`DDIMScheduler` or `LCMScheduler`):
            Scheduler used to perform the diffusion denoising steps.
        text_encoder (`CLIPTextModel`):
            Text encoder used to produce the (empty) text embedding passed to the UNet.
        tokenizer (`CLIPTokenizer`):
            Tokenizer used to build inputs for the text encoder.
        default_denoising_steps (`int`, *optional*):
            Default number of denoising steps used when the pipeline is called without an explicit value.
        default_processing_resolution (`int`, *optional*):
            Default maximum edge resolution used to resize inputs when the pipeline is called without an explicit value.

    Output:
        The pipeline returns a `MarigoldGSFixerOutput` with:
          - fixed_rgb: a PIL.Image.Image (uint8, [0,255]) of the fixed RGB result.
          - fixed_rgb_ts: a torch.Tensor (float, [0.0,1.0]) containing the fixed RGB result.

    """

    latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        condition_image1: Union[Image.Image, torch.Tensor],
        condition_image2: Optional[Union[Image.Image, torch.Tensor]] = None,
        denoising_steps: Optional[int] = None,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = False,
    ) -> MarigoldGSFixerOutput:
        """
        Invoke the Marigold GSFixer pipeline.

        Args:
            condition_image1 (PIL.Image.Image or torch.Tensor):
            First conditioning RGB image (gs). If PIL.Image is provided it will be converted to RGB and to a tensor
            of shape [1, 3, H, W].
            condition_image2 (PIL.Image.Image or torch.Tensor, optional):
            Optional second conditioning RGB image (mesh). If provided, must have the same shape as condition_image1.
            denoising_steps (int, optional):
            Number of diffusion denoising steps to run. If None, the pipeline default is used.
            processing_res (int, optional):
            Maximum edge resolution used to resize inputs prior to processing. If None, the pipeline default is used.
            If set to 0, the inputs are processed at their original resolution.
            match_input_res (bool, optional):
            If True (default) resize the final prediction back to the original input resolution.
            resample_method (str, optional):
            Resampling method used when resizing images. One of 'bilinear', 'bicubic' or 'nearest'. Default 'bilinear'.
            batch_size (int, optional):
            Inference batch size for internal batching. If 0 (default) the pipeline will pick an appropriate value.
            generator (torch.Generator or None, optional):
            Random generator used for initial noise sampling.
            show_progress_bar (bool, optional):
            Whether to display progress bars for batching and denoising. Default False.

        Returns:
            MarigoldGSFixerOutput:
            - fixed_rgb: PIL.Image.Image or None, final fixed RGB image (uint8, [0,255]).
            - fixed_rgb_ts: torch.Tensor, float tensor of the fixed RGB image in range [0.0, 1.0].
        """
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(condition_image1, Image.Image):
            condition_image1 = condition_image1.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            condition_rgb1 = pil_to_tensor(condition_image1)
            condition_rgb1 = condition_rgb1.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(condition_image1, torch.Tensor):
            condition_rgb1 = condition_image1
        else:
            raise TypeError(f"Unknown input type: {type(condition_image1) = }")
        input_size = condition_rgb1.shape
        assert (
            condition_rgb1.dim() == 4 and input_size[-3] == 3
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        if condition_image2 is not None:
            if isinstance(condition_image2, Image.Image):
                condition_image2 = condition_image2.convert("RGB")
                # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
                condition_rgb2 = pil_to_tensor(condition_image2)
                condition_rgb2 = condition_rgb2.unsqueeze(0)  # [1, rgb, H, W]
            elif isinstance(condition_image2, torch.Tensor):
                condition_rgb2 = condition_image2
            else:
                raise TypeError(f"Unknown input type: {type(condition_image2) = }")
            assert (
                condition_rgb2.shape == input_size
            ), f"Mismatched shape {condition_rgb2.shape} and {input_size}"

        # Resize image
        if processing_res > 0:
            condition_rgb1 = resize_max_res(
                condition_rgb1,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

            if condition_image2 is not None:
                condition_rgb2 = resize_max_res(
                    condition_rgb2,
                    max_edge_resolution=processing_res,
                    resample_method=resample_method,
                )

        # Normalize rgb values
        condition_rgb1_norm: torch.Tensor = condition_rgb1 / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        condition_rgb1_norm = condition_rgb1_norm.to(self.dtype)
        assert condition_rgb1_norm.min() >= -1.0 and condition_rgb1_norm.max() <= 1.0

        if condition_image2 is not None:
            condition_rgb2_norm: torch.Tensor = condition_rgb2 / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
            condition_rgb2_norm = condition_rgb2_norm.to(self.dtype)
            assert condition_rgb2_norm.min() >= -1.0 and condition_rgb2_norm.max() <= 1.0

        # ----------------- Predicting -----------------
        # Batch repeated input image
        duplicated_condition_rgb1 = condition_rgb1_norm.expand(1, -1, -1, -1)
        if condition_image2 is not None:
            duplicated_condition_rgb2 = condition_rgb2_norm.expand(1, -1, -1, -1)
            single_rgb_dataset = TensorDataset(duplicated_condition_rgb1, duplicated_condition_rgb2)
        else:
            single_rgb_dataset = TensorDataset(duplicated_condition_rgb1)

        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=1,
                input_res=max(condition_rgb1_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict (batched)
        target_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader

        for batch in iterable:
            if condition_image2 is not None:
                (batched_img1,batched_img2,) = batch
                target_pred_raw = self.single_infer(
                    condition_rgb1_in=batched_img1,
                    condition_rgb2_in=batched_img2,
                    num_inference_steps=denoising_steps,
                    show_pbar=show_progress_bar,
                    generator=generator,
                )
            else:
                (batched_img1,) = batch
                target_pred_raw = self.single_infer(
                    condition_rgb1_in=batched_img1,
                    num_inference_steps=denoising_steps,
                    show_pbar=show_progress_bar,
                    generator=generator,
                )

            target_pred_ls.append(target_pred_raw.detach())
        
        target_preds = torch.concat(target_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache       
        final_pred = target_preds

        # Resize back to original resolution
        if match_input_res:
            final_pred = resize(
                final_pred,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )

        # Convert to numpy
        final_pred = final_pred.squeeze().cpu()

        final_pred_img = (final_pred.numpy() * 255).astype(np.uint8)
        final_pred_img = chw2hwc(final_pred_img)
        final_pred_img = Image.fromarray(final_pred_img)

        return MarigoldGSFixerOutput(
            fixed_rgb=final_pred_img,
            fixed_rgb_ts=final_pred,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if "trailing" != self.scheduler.config.timestep_spacing:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `timestep_spacing="
                    f'"{self.scheduler.config.timestep_spacing}"`; the recommended setting is `"trailing"`. '
                    f"This change is backward-compatible and yields better results. "
                    f"Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
                )
            else:
                if n_step > 10:
                    logging.warning(
                        f"Setting too many denoising steps ({n_step}) may degrade the prediction; consider relying on "
                        f"the default values."
                    )
            if not self.scheduler.config.rescale_betas_zero_snr:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `rescale_betas_zero_snr="
                    f"{self.scheduler.config.rescale_betas_zero_snr}`; the recommended setting is True. "
                    f"Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            logging.warning(
                "DeprecationWarning: LCMScheduler will not be supported in the future. "
                "Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
            )
            if n_step > 10:
                logging.warning(
                    f"Setting too many denoising steps ({n_step}) may degrade the prediction; consider relying on "
                    f"the default values."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        condition_rgb1_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        condition_rgb2_in: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Perform a single prediction with the diffusion model.

        Args:
            condition_rgb1_in (torch.Tensor):
            Batched input RGB tensor (gs) for the primary condition (shape [B, 3, H, W]),
            expected in the pipeline dtype and normalized to [-1, 1].
            num_inference_steps (int):
            Number of diffusion denoising steps to run.
            generator (torch.Generator or None):
            Optional torch generator used for initial noise sampling.
            show_pbar (bool):
            Whether to display a progress bar during the denoising loop.
            condition_rgb2_in (torch.Tensor, optional):
            Optional batched input RGB tensor (mesh) for the second condition (same shape as
            condition_rgb1_in). If provided, it is used together with condition_rgb1_in
            as additional conditioning for the UNet.

        Returns:
            torch.Tensor: Predicted RGB tensor in [0, 1] of shape [B, 3, H, W].
        """
        device = self.device
        condition_rgb1_in = condition_rgb1_in.to(device)
        condition_rgb1_latent = self.encode_rgb(condition_rgb1_in)  # [B, 4, h, w]
        if condition_rgb2_in is not None:
            condition_rgb2_in = condition_rgb2_in.to(device)
            condition_rgb2_latent = self.encode_rgb(condition_rgb2_in)  # [B, 4, h, w]

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Noisy latent for outputs
        target_latent = torch.randn(
            condition_rgb1_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (condition_rgb1_latent.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            if condition_rgb2_in is not None:
                unet_input = torch.cat(
                    [condition_rgb2_latent, condition_rgb1_latent, target_latent], dim=1
                )  # the concat order is [mesh_latent, gs_latent, target_latent]
            else:
                unet_input = torch.cat(
                    [condition_rgb1_latent, target_latent], dim=1
                )

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            target_latent = self.scheduler.step(
                noise_pred, t, target_latent, generator=generator
            ).prev_sample
        
        rgb = self.decode_rgb(target_latent)  # [B,3,H,W]

        # clip prediction
        rgb = torch.clip(rgb, -1.0, 1.0)
        # shift to [0, 1]
        rgb = (rgb + 1.0) / 2.0

        return rgb

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.latent_scale_factor
        return rgb_latent
    
    def decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode rgb latent into rgb image.

        Args:
            rgb_latent (`torch.Tensor`):
                RGB latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded rgb image.
        """
        # scale latent
        rgb_latent = rgb_latent / self.latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(rgb_latent)
        rgb = self.vae.decoder(z)
        return rgb
