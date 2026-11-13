# Vendored DOVE degradation

Source: https://github.com/zhengchen1999/DOVE (commit `1a499a8`), copied verbatim:

- `degradation.py`  ← `finetune/datasets/degradation.py` (OpenMMLab header, Apache-2.0)
- `blur_kernels.py` ← `finetune/datasets/blur_kernels.py` (BasicSR-derived, Apache-2.0, © 2020 xinntao)
- `degradation.yaml` ← `finetune/configs/degradation.yaml`

This is the **RealESRGAN / RealBasicVSR second-order random degradation** DOVE uses to
synthesize LQ video from HQ (blur → resize → noise → JPEG → video-compression, run
twice, then a shuffled {video-compression | [resize-to-target, sinc-blur]} block),
at **×4**. It is self-contained — deps: `numpy, scipy, opencv-python, av (PyAV), pyyaml`
(no basicsr / realesrgan / torch for the pixel degradation).

Orchestration (first_stage / second_stage, target_size = input H/4,W/4) is reproduced in
`../make_lq.py`, matching `finetune/datasets/real_sr_dataset.py::RealSRDataset.degrade`.
Used here only for LQ generation in the reduced-reference consistency benchmark.

**Local edit (behavior-preserving):** in `degradation.py::RandomVideoCompression`,
`frame.pict_type = 'NONE'` is wrapped in try/except so it also works on PyAV ≥ ~10
(which requires `PictureType.NONE` instead of the string). `NONE` = "encoder decides"
in both; the encoded output is unchanged. This is the only modification to the sources.
