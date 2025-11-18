<h1 align="center"> GSFix3D: Diffusion-Guided Repair of Novel Views in Gaussian Splatting </h1>

<h3 align="center"> Jiaxin Wei, Stefan Leutenegger, Simon Schaefer </h3>

<h3 align="center">
  <a href="https://arxiv.org/pdf/2508.14717">Paper</a> | <a href="https://youtu.be/hF8xv8qDSi0">Video</a> | <a href="https://gsfix3d.github.io/">Project Page</a> | <a href="https://huggingface.co/collections/goldoak1421/gsfix3d">Model</a>
</h3>

<p align="center">
  <a href="">
    <img src="./media/teaser.gif" alt="teaser" width="100%">
  </a>
</p>

<p align="center"> TL;DR: Remove artifacts and fill holes for novel views in 3DGS scenes. </p>

Abstract: *Recent developments in 3D Gaussian Splatting have significantly enhanced novel view synthesis, yet generating high-quality renderings from extreme novel viewpoints or partially observed regions remains challenging. Meanwhile, diffusion models exhibit strong generative capabilities, but their reliance on text prompts and lack of awareness of specific scene information hinder accurate 3D reconstruction tasks. To address these limitations, we introduce GSFix3D, a novel framework that improves the visual fidelity in under-constrained regions by distilling prior knowledge from diffusion models into 3D representations, while preserving consistency with observed scene details. At its core is GSFixer, a latent diffusion model obtained via our customized fine-tuning protocol that can leverage both mesh and 3D Gaussians to adapt pretrained generative models to a variety of environments and artifact types from different reconstruction methods, enabling robust novel view repair for unseen camera poses. Moreover, we propose a random mask augmentation strategy that empowers GSFixer to plausibly inpaint missing regions. Experiments on challenging benchmarks demonstrate that our GSFix3D and GSFixer achieve state-of-the-art performance, requiring only minimal scene-specific fine-tuning on captured data. Real-world test further confirms its resilience to potential pose errors.*

## News
- **[2025-11-18]**: Released the code and data of GSFix3D. More models are coming soon, stay tuned!
- **[2025-11-06]**: Our paper has been accepted by International Conference on 3D Vision 2026 (3DV)!
- **[2025-08-20]**: Released the paper on arXiv.

## Setup

**We tested our code on: Ubuntu 22.04 LTS, Python 3.11.14, CUDA 12.1, NVIDIA RTX 4500 Ada.**

Clone the repository with `--recursive` flag
```
git clone https://github.com/GSFix3D/GSFix3D.git --recursive
```

Create virtual environment
```
cd GSFix3D
conda create -n gsfix3d python=3.11
conda activate gsfix3d
pip install -r requirements.txt
```

Build differentiable Gaussian rasterizer
```
cd diff-gaussian-rasterization
pip install .
```

## Usage
### Training from scratch
You can skip training GSFixer from scratch and directly use our pretrained [checkpoints](https://hf.co/collections/goldoak1421/gsfix3d) and proceed with fine-tuning in the next step. Or you can download Stable Diffusion v2 [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2) into `${BASE_CKPT_DIR}` and use the provided training script to train it from scratch. **We notice that the official Stable Diffusion v2 has been removed from Huggingface, but it may still be accessible through third-party sources or other community-maintained archives.** Please refer to [here](https://github.com/prs-eth/Marigold/tree/main?tab=readme-ov-file#prepare-for-training-data) for training data preparation.

Set environment parameters for the data directory
```
export BASE_DATA_DIR=YOUR_DATA_DIR        # directory to training data
export BASE_CKPT_DIR=YOUR_CHECKPOINT_DIR  # directory to pretrained checkpoint
```

Adjust hyperparameters in YAML files under `config/train_gsfixer` to determine whether to train with or without dual input
```
dual_input: true  # or false to train GSFixer with only gs input
```
You also need to adjust the checkpoint name in `config/model_sdv2.yaml` if you are using an unofficial model with a different name
```
pretrained_path: stable-diffusion-2  # checkpoint name in ${BASE_CKPT_DIR}
```

Run the training script
```
python scripts/gsfixer/train.py --config config/train_gsfixer.yaml --output_dir <output_path>
```
Note that only `UNet` and `scheduler` will be saved in the `--output_dir`. You need to reuse other model components from the pretrained checkpoint in `BASE_CKPT_DIR`.

### Fine-tuning on captured data
Now you can fine-tune the base GSFixer model on any captured data. First, you need to organize the captured data as follows (here we take the `room1` scene from the Replica dataset for example). Specifically, you need to put rendered images from the 3DGS model and (optionally) mesh, as well as the captured ground truth RGB images, under a `finetune` folder to construct data pairs for fine-tuning. Please refer to [here](https://github.com/ethz-mrl/GSFusion#download-datasets) for downloading the Replica and ScanNet++ datasets. We also provide the rendered images from [GSFusion](https://github.com/ethz-mrl/GSFusion), [SplaTAM](https://github.com/spla-tam/SplaTAM) and [RTG-SLAM](https://github.com/MisEty/RTG-SLAM) for Replica and ScanNet++ datasets, which can be downloaded [here](https://drive.google.com/drive/folders/1bpOWQAcSH-jv9geZCsWyC8TdwM4gYLGJ?usp=sharing). Note that `novel_views` and `novel_views.json` are the novel view data we created for quantitative evaluation on the Replica dataset. Please refer to our paper for more details on the novel view selection process.
```
Replica
├── cam_params.json
└── room1
    ├── finetune
    |   ├── rendered_gs
    |   ├── rendered_mesh
    |   └── rgb
    ├── novel_views
    ├── novel_views.json
    ├── rendered_novel_views
    |   ├── gs_image_rtg
    |   ├── gs_image_splatam
    |   ├── gs_image_gsfusion
    |   └── mesh_image
    ├── results
    └── traj.txt
```

Set environment parameters for the data directory
```
export BASE_DATA_DIR=CAPTURED_DATA_DIR        # directory to captured data
export BASE_CKPT_DIR=GSFIXER_CHECKPOINT_DIR   # directory to pretrained GSFixer checkpoint
```

Adjust hyperparameters in YAML files under `config/finetune_gsfixer`
```
dual_input: true     # or false to fine-tune GSFixer with only gs input
dir: room1/finetune  # path to fine-tuning data
```
You also need to adjust the checkpoint name in `config/model_gsfixer.yaml` if you are using your own GSFixer checkpoint with a different name. Note that the checkpoint in use must be consistent with the dual_input flag set in the YAML file above.
```
pretrained_path: gsfixer-full # checkpoint name in ${BASE_CKPT_DIR}
```

Run the fine-tuning script
```
python scripts/gsfixer/train.py --config config/finetune_gsfixer.yaml --output_dir <output_path>
```
Note that only `UNet` and `scheduler` will be saved in the `--output_dir`. You need to reuse other model components from the pretrained GSFixer checkpoint in `BASE_CKPT_DIR`.

### Inference with GSFixer
Run the inference script to fix 2D novel views using the fine-tuned GSFixer model
```
python scripts/gsfixer/inference.py --checkpoint <diffusion_model_path> --recon_method_type <recon_method_name> --data_type <supported_data_type> --data_path <captured_data_path> --output_dir <output_path>  --dual_input --eval
```
**The fine-tuned GSFixer checkpoints for the Replica and ScanNet++ datasets will be available soon on Hugging Face.** If you are testing on Replica or ScanNet++ dataset, the `<captured_data_path>` is the path to the organized dataset detailed above and you can use `--eval` flag to enable evaluation on repaired novel views obtained from GSFixer; otherwise, you need to provide the path to your captured data. `--recon_method_type` is used to locate input (rendered) gs images with corresponding file name postfix. You can enable dual-conditioning by using `--dual_input` flag.

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for inference.py</span></summary>

  #### --checkpoint <diffusion_model_path>
  Path that contains the fine-tuned GSFixer model or the Hugging Face model ID.
  #### --recon_method_type <recon_method_name>
  File name postfix to locate input gs images. We provide reconstruction data from `gsfusion`, `splatam` and `rtg`.
  #### --data_type <supported_data_type>
  Currently supported data types are `replica` and `scannetpp`.
  #### --data_path <captured_data_path>
  Path to your captured data.
  #### --dual_input
  Enable dual-conditioning (mesh+gs).
  #### --eval
  Enable evaluation on fixed novel views obtained from GSFixer. Note that this flag is only valid for the Replica dataset or ScanNet++ dataset. For your own data, you need to prepare the ground truth for novel views to use this function.

</details>

### Lifting to 3D with GSFix3D
We provide a script to lift the results of GSFixer back into the 3DGS model
```
python scripts/gsfix3d/refine_gs.py -m <gs_model_path> --data_type <supported_data_type> --data_path <captured_data_path> --fixed_image_path <gsfixer_results_path> --output_dir <output_path> --eval
```
Please refer to [here](https://github.com/GS-Fusion/GSFusion_eval#file-structure) for the expected file structure of `<gs_model_path>`. We provide several example 3DGS maps [here](https://cloud.cvai.cit.tum.de/s/feg33Y8wGMGEC9t) for the Replica and ScanNet++ datasets. If you are testing on Replica or ScanNet++ dataset, the `<captured_data_path>` is the path to the organized dataset above and you can use `--eval` flag to enable evaluation on novel views rendered from the refined 3DGS model; otherwise, you need to provide the path to your captured data and `--novel_views` the novel view camera poses in `json` file format. Please refer to the next section for more details about how to prepare your own data.

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for refine_gs.py</span></summary>

  #### -m <gs_model_path>
  Path that contains the 3DGS model.
  #### --data_type <supported_data_type>
  Currently supported data types are `replica` and `scannetpp`.
  #### --data_path <captured_data_path>
  Path to your captured data.
  #### --sparse_kf_list (optional)
  A keyframe list to only select keyframes from ground truth data for further refinement of the 3DGS model, which is recommended for reducing optimization time.
  #### --fixed_image_path <gsfixer_results_path>
  Path to fixed novel view images obtained from GSFixer.
  #### --novel_views
  JSON file path to novel view camera poses.
  #### --eval
  Enable evaluation on novel views rendered from the refined 3DGS model. Note that this flag is only valid for the Replica dataset or ScanNet++ dataset. For your own data, you need to prepare the ground truth for novel views to use this function.

</details>

## Testing on your own data

Prepare the novel views that you want to repair using our provided script
```
python scripts/gsfix3d/novel_view_capture.py -m <gs_model_path> --data_type <supported_data_type> --data_path <customized_data_path> --output_dir <output_path> --render_mesh
```
Here we only provide a simple script to simplify the process of novel view selection and rendering from selected novel views. Note that our script requires loading a mesh model for interaction in Open3D. The mesh is also needed if you want to render novel view images from the mesh. You can download several example mesh files [here](https://cloud.cvai.cit.tum.de/s/feg33Y8wGMGEC9t) for the Replica and ScanNet++ datasets or run [GSFusion](https://github.com/ethz-mrl/GSFusion) to obtain both mesh and 3DGS map simultaneously. You can also use other methods (eliminating the requirement for a mesh model) to obtain novel view data (i.e., poses and rendered images), as long as you follow the same format as ours.

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for novel_view_capture.py</span></summary>

  #### -m <gs_model_path>
  Path that contains the 3DGS model and mesh. The expected file structure of `<gs_model_path>` is explained [here](https://github.com/GS-Fusion/GSFusion_eval/tree/main#file-structure).
  #### --data_type <supported_data_type>
  Currently supported data types are `replica` and `scannetpp`. We highly recommend that you organize your customized data in Replica format so that it can be directly recognized by our provided scripts.
  #### --data_path <customized_data_path>
  Path to your customized data.
  #### --render_mesh
  Enable mesh rendering from selected novel views.

</details>


## Citation

If you find our work useful, please cite us:
```bibtex
@article{wei2026gsfix3d,
  title={GSFix3D: Diffusion-Guided Repair of Novel Views in Gaussian Splatting}, 
  author={Wei, Jiaxin and Leutenegger, Stefan and Schaefer, Simon},
  booktitle={2026 International Conference on 3D Vision (3DV)},
  year={2026},
  organization={IEEE}
}
```


## License

Copyright (c) 2025, Jiaxin Wei

### Important Notice
- This project includes code licensed under the [Apache License Version 2.0](./LICENSES/Apache-2.0.txt). 
- This project includes components licensed under the [Gaussian-Splatting-License](./LICENSES/Gaussian-Splatting-License.md), which restricts the entire project to **non-commercial use only**. If you wish to use this project for commercial purposes, please contact the respective copyright holders for permission.
- The provided models are licensed under [RAIL++-M License](./LICENSES/LICENSE-MODEL.txt).
- Users must comply with all license requirements included in this repository.


## Acknowledgement
The authors gratefully acknowledge support from the EU project AUTOASSESS (Grant 101120732). We also thank Jaehyung Jung and Sebastián Barbas Laina for their assistance with ship data collection and processing, and Helen Oleynikova for her valuable feedback on the manuscript.