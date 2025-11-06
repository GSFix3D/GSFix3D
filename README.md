<h1 align="center"> GSFix3D: Diffusion-Guided Repair of Novel Views in Gaussian Splatting </h1>

<h3 align="center"> Jiaxin Wei, Stefan Leutenegger, Simon Schaefer </h3>

<h3 align="center">
  <a href="https://arxiv.org/pdf/2508.14717">Paper</a> | <a href="https://youtu.be/hF8xv8qDSi0">Video</a> | <a href="https://gsfix3d.github.io/">Project Page</a>
</h3>

<p align="center">
  <a href="">
    <img src="./media/teaser.gif" alt="teaser" width="100%">
  </a>
</p>

<p align="center"> TL;DR: Remove artifacts and fill holes for novel views in 3DGS scenes. </p>

Abstract: *Recent developments in 3D Gaussian Splatting have significantly enhanced novel view synthesis, yet generating high-quality renderings from extreme novel viewpoints or partially observed regions remains challenging. Meanwhile, diffusion models exhibit strong generative capabilities, but their reliance on text prompts and lack of awareness of specific scene information hinder accurate 3D reconstruction tasks. To address these limitations, we introduce GSFix3D, a novel framework that improves the visual fidelity in under-constrained regions by distilling prior knowledge from diffusion models into 3D representations, while preserving consistency with observed scene details. At its core is GSFixer, a latent diffusion model obtained via our customized fine-tuning protocol that can leverage both mesh and 3D Gaussians to adapt pretrained generative models to a variety of environments and artifact types from different reconstruction methods, enabling robust novel view repair for unseen camera poses. Moreover, we propose a random mask augmentation strategy that empowers GSFixer to plausibly inpaint missing regions. Experiments on challenging benchmarks demonstrate that our GSFix3D and GSFixer achieve state-of-the-art performance, requiring only minimal scene-specific fine-tuning on captured data. Real-world test further confirms its resilience to potential pose errors.*

## News
- **[2025-11-06]**: Our paper has been accepted by International Conference on 3D Vision 2026 (3DV)!
- **[2025-08-20]**: Released the paper on arXiv.


## Citation

If you find our work useful, please cite us:
```bibtex
@article{gsfix3d,
         title={GSFix3D: Diffusion-Guided Repair of Novel Views in Gaussian Splatting}, 
         author={Jiaxin Wei and Stefan Leutenegger and Simon Schaefer},
         year={2025},
         eprint={2508.14717},
         archivePrefix={arXiv},
         primaryClass={cs.CV},
         url={https://arxiv.org/abs/2508.14717},
}
```


## Acknowledgement
The authors gratefully acknowledge support from the EU project AUTOASSESS (Grant 101120732). We also thank Jaehyung Jung and Sebastián Barbas Laina for their assistance with ship data collection and processing, and Helen Oleynikova for her valuable feedback on the manuscript.