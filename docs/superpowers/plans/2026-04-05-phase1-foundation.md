# Phase 1: Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete baseline verification, run baselines on long videos to document failure modes, write Introduction and Literature Review chapter drafts for VSR thesis.

**Architecture:** Three parallel workstreams: (1) server-side baseline completion (UAV evaluation + long-video tests), (2) literature review research, (3) thesis chapter writing. Chapters reuse zjuthesis template structure — rewrite `introduction.tex` and `literature-review.tex` in place. BibTeX entries go in `ref.bib`.

**Tech Stack:** LaTeX (XeLaTeX via latexmk), Python (pyiqa), SSH to GPU server, BibTeX

**Spec:** `docs/superpowers/specs/2026-04-05-thesis-completion-plan-design.md` (Phase 1 section)

---

## File Map

**Modify:**
- `zjuthesis/body/graduate-eng/introduction.tex` — Full rewrite for VSR topic
- `zjuthesis/body/graduate-eng/literature-review.tex` — Full rewrite, currently template placeholder
- `zjuthesis/body/ref.bib` — Replace VOS references with VSR references
- `zjuthesis/zjuthesis.tex` — Update Title, TitleEng, Topic fields
- `experiments/baselines/target_metrics.md` — Update with UAV verified results

**Create:**
- `docs/research/vsr-literature-notes.md` — Structured research notes from deep literature survey
- `experiments/baselines/long_video_eval/` — Directory for long-video baseline failure analysis
- `experiments/baselines/long_video_eval/README.md` — Document long-video test methodology and findings

**No new Python code needed** — existing `evaluate_pyiqa.py` and `evaluate_pyiqa_nr.py` handle all evaluation.

---

### Task 1: Complete UAV baseline verification (server-side) — DONE

**Files:**
- Modify: `experiments/baselines/target_metrics.md`

Completed April 6-7. UAV UDM10 inference finished. Evaluated and found significant gap from paper (PSNR 24.94 vs 30.79). Investigated with param sweep and DOVE cross-validation — confirmed pipeline is correct, gap is degradation mismatch.

- [x] **Step 1: Check UAV inference status on server** — Done. Inference completed.
- [x] **Step 2: Evaluate UAV UDM10 results with pyiqa** — Done. Result: PSNR 24.94, SSIM 0.7085, LPIPS 0.3280.
- [x] **Step 3: Copy results to local machine** — Done. `uav_UDM10_rbvsr_pyiqa.json` copied.
- [x] **Step 4: Compare with paper targets and update target_metrics.md** — Done. Significant gap: PSNR 24.94 vs paper 30.79 (-5.85 dB). Investigated with param sweep (5 configs, ~0.9 dB variation — not hyperparameters) and DOVE cross-validation (pipeline verified). Root cause: degradation mismatch.
- [x] **Step 5: Commit** — Done. Commit `c4bedc3`.

---

### Task 2: Run UAV on YouHQ40 with RealBasicVSR degradation (server-side) — DONE

**Files:**
- Modify: `experiments/baselines/target_metrics.md`

Completed April 6-7. Results: PSNR 23.40 vs paper 25.83 (-2.43 dB gap). Same degradation mismatch as UDM10. Results copied locally and committed.

- [x] **Step 1: Check GPU availability on server** — Done.
- [x] **Step 2: Launch UAV YouHQ40 inference** — Done.
- [x] **Step 3: Evaluate and copy results back** — Done. `uav_YouHQ40_rbvsr_default_pyiqa.json` copied.
- [x] **Step 4: Update target_metrics.md and commit** — Done. Commit `c4bedc3`.

---

### Task 3: Update zjuthesis metadata for VSR topic — NOT STARTED

**Files:**
- Modify: `zjuthesis/zjuthesis.tex`

Note: Title is preliminary — will be refined once method is confirmed after April 9 meeting. Defer until direction is clearer.

- [ ] **Step 1: Update thesis metadata fields**
- [ ] **Step 2: Verify it compiles**
- [ ] **Step 3: Commit**

---

### Task 4: Replace ref.bib with VSR references — REMOVED

Replaced by incremental approach: bibliography entries will be added as chapters are written, not pre-populated in bulk. The BibTeX entries listed below are kept as reference for when writing begins.

**Files:**
- Modify: `zjuthesis/body/ref.bib`

Reference BibTeX entries for VSR thesis:

```bibtex
% === Video Super-Resolution: CNN-based ===

@inproceedings{chan_basicvsr_2021,
  title     = {{BasicVSR}: The Search for Essential Components in Video Super-Resolution and Their Beyond},
  author    = {Chan, Kelvin C.K. and Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
  booktitle = {CVPR},
  year      = {2021}
}

@inproceedings{chan_basicvsrpp_2022,
  title     = {{BasicVSR++}: Improving Video Super-Resolution with Enhanced Propagation and Alignment},
  author    = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  booktitle = {CVPR},
  year      = {2022}
}

@inproceedings{wang_edvr_2019,
  title     = {{EDVR}: Video Restoration with Enhanced Deformable Convolutional Networks},
  author    = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
  booktitle = {CVPRW},
  year      = {2019}
}

@inproceedings{chan_realbasicvsr_2022,
  title     = {{RealBasicVSR}: Investigating Tradeoffs in Real-World Video Super-Resolution},
  author    = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  booktitle = {CVPR},
  year      = {2022}
}

% === Video Super-Resolution: Transformer-based ===

@inproceedings{liang_vrt_2022,
  title     = {{VRT}: A Video Restoration Transformer},
  author    = {Liang, Jingyun and Cao, Jiezhang and Fan, Yuchen and Zhang, Kai and Ranjan, Rakesh and Li, Yawei and Timofte, Radu and Van Gool, Luc},
  booktitle = {TIP},
  year      = {2024}
}

@inproceedings{liang_rvrt_2024,
  title     = {Recurrent Video Restoration Transformer with Guided Deformable Attention},
  author    = {Liang, Jingyun and Fan, Yuchen and Xiang, Xiaoyu and Ranjan, Rakesh and Ilg, Eddy and Green, Simon and Cao, Jiezhang and Zhang, Kai and Timofte, Radu and Van Gool, Luc},
  booktitle = {NeurIPS},
  year      = {2024}
}

@inproceedings{shi_rethinking_psrt_2022,
  title     = {Rethinking Alignment in Video Super-Resolution Transformers},
  author    = {Shi, Shuwei and Gu, Jinjin and Xie, Liangbin and Wang, Xintao and Yang, Yujiu and Dong, Chao},
  booktitle = {NeurIPS},
  year      = {2022}
}

% === Video Super-Resolution: Diffusion-based ===

@inproceedings{zhou_upscale_a_video_2024,
  title     = {Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution},
  author    = {Zhou, Shangchen and Yang, Peiqing and Wang, Jianyi and Luo, Yihang and Loy, Chen Change},
  booktitle = {CVPR},
  year      = {2024}
}

@inproceedings{yeung_mgldvsr_2024,
  title     = {Motion-Guided Latent Diffusion for Temporally Consistent Real-World Video Super-Resolution},
  author    = {Yeung, Xi and Chang, Huiqiang and Li, Zhongwei and Li, Bing},
  booktitle = {ECCV},
  year      = {2024}
}

@inproceedings{wang_stablesr_2024,
  title     = {Exploiting Diffusion Prior for Real-World Image Super-Resolution},
  author    = {Wang, Jianyi and Yue, Zongsheng and Zhou, Shangchen and Chan, Kelvin C.K. and Loy, Chen Change},
  booktitle = {IJCV},
  year      = {2024}
}

@inproceedings{yang_mgldvsr_2024,
  title     = {Motion-Guided Latent Diffusion for Temporally Consistent Real-world Video Super-Resolution},
  author    = {Yang, Xi and Chang, Huiqiang and Li, Zhongwei and Li, Bing},
  booktitle = {ECCV},
  year      = {2024}
}

% === Image Super-Resolution foundations ===

@inproceedings{dong_srcnn_2014,
  title     = {Learning a Deep Convolutional Network for Image Super-Resolution},
  author    = {Dong, Chao and Loy, Chen Change and He, Kaiming and Tang, Xiaoou},
  booktitle = {ECCV},
  year      = {2014}
}

@inproceedings{lim_edsr_2017,
  title     = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  author    = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  booktitle = {CVPRW},
  year      = {2017}
}

@inproceedings{zhang_rcan_2018,
  title     = {Image Super-Resolution Using Very Deep Residual Channel Attention Networks},
  author    = {Zhang, Yulun and Li, Kunpeng and Li, Kai and Wang, Lichen and Zhong, Bineng and Fu, Yun},
  booktitle = {ECCV},
  year      = {2018}
}

@inproceedings{liang_swinir_2021,
  title     = {{SwinIR}: Image Restoration Using Swin Transformer},
  author    = {Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle = {ICCVW},
  year      = {2021}
}

% === Diffusion Models ===

@inproceedings{ho_ddpm_2020,
  title     = {Denoising Diffusion Probabilistic Models},
  author    = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle = {NeurIPS},
  year      = {2020}
}

@inproceedings{rombach_ldm_2022,
  title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle = {CVPR},
  year      = {2022}
}

@inproceedings{song_ddim_2021,
  title     = {Denoising Diffusion Implicit Models},
  author    = {Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  booktitle = {ICLR},
  year      = {2021}
}

% === State-Space Models ===

@inproceedings{gu_s4_2022,
  title     = {Efficiently Modeling Long Sequences with Structured State Spaces},
  author    = {Gu, Albert and Goel, Karan and R{\'e}, Christopher},
  booktitle = {ICLR},
  year      = {2022}
}

@article{gu_mamba_2023,
  title     = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author    = {Gu, Albert and Dao, Tri},
  journal   = {arXiv preprint arXiv:2312.00752},
  year      = {2023}
}

@article{dao_mamba2_2024,
  title     = {Transformers are {SSMs}: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author    = {Dao, Tri and Gu, Albert},
  journal   = {arXiv preprint arXiv:2405.21060},
  year      = {2024}
}

@article{po_longcontext_ssm_2025,
  title     = {Long-Context State-Space Video World Models},
  author    = {Po, Ryan and Nitzan, Yael and Zhang, Richard and Chen, Haojie and Dao, Tri and Shechtman, Eli and Wetzstein, Gordon and Huang, De-An},
  journal   = {arXiv preprint arXiv:2505.20171},
  year      = {2025}
}

% === SSM in Vision ===

@inproceedings{zhu_vmamba_2024,
  title     = {Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author    = {Zhu, Lianghui and Liao, Bencheng and Zhang, Qian and Wang, Xinlong and Liu, Wenyu and Wang, Xinggang},
  booktitle = {ICML},
  year      = {2024}
}

@article{shi_vmambair_2024,
  title     = {{VMambaIR}: Visual State Space Model for Image Restoration},
  author    = {Shi, Yuan and Xia, Bin and Jin, Xiaoyu and Wang, Xing and Zhao, Tianyu and Xia, Xin and Xiao, Xuefeng and Yang, Wenming},
  journal   = {arXiv preprint arXiv:2403.11423},
  year      = {2024}
}

% === Optical Flow / Alignment ===

@inproceedings{ranjan_spynet_2017,
  title     = {Optical Flow Estimation Using a Spatial Pyramid Network},
  author    = {Ranjan, Anurag and Black, Michael J.},
  booktitle = {CVPR},
  year      = {2017}
}

@inproceedings{teed_raft_2020,
  title     = {{RAFT}: Recurrent All-Pairs Field Transforms for Optical Flow},
  author    = {Teed, Zachary and Deng, Jia},
  booktitle = {ECCV},
  year      = {2020}
}

% === Evaluation / Metrics ===

@article{wang_ssim_2004,
  title     = {Image Quality Assessment: From Error Visibility to Structural Similarity},
  author    = {Wang, Zhou and Bovik, Alan C. and Sheikh, Hamid R. and Simoncelli, Eero P.},
  journal   = {IEEE TIP},
  volume    = {13},
  number    = {4},
  pages     = {600--612},
  year      = {2004}
}

@inproceedings{zhang_lpips_2018,
  title     = {The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author    = {Zhang, Richard and Isola, Phillip and Efros, Alexei A. and Shechtman, Eli and Wang, Oliver},
  booktitle = {CVPR},
  year      = {2018}
}

% === Benchmarks / Datasets ===

@inproceedings{yi_udm10_2019,
  title     = {Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations},
  author    = {Yi, Peng and Wang, Zhongyuan and Jiang, Kui and Jiang, Junjun and Ma, Jiayi},
  booktitle = {ICCV},
  year      = {2019}
}

@inproceedings{nah_reds_2019,
  title     = {{NTIRE} 2019 Challenge on Video Deblurring and Super-Resolution: Dataset and Study},
  author    = {Nah, Seungjun and Baik, Sungyong and Hong, Seokil and Moon, Gyeongsik and Son, Sanghyun and Timofte, Radu and Lee, Kyoung Mu},
  booktitle = {CVPRW},
  year      = {2019}
}

@inproceedings{liu_vid4_2013,
  title     = {On {Bayesian} Adaptive Video Super Resolution},
  author    = {Liu, Ce and Sun, Deqing},
  booktitle = {IEEE TPAMI},
  year      = {2014}
}

@inproceedings{xue_vimeo90k_2019,
  title     = {Video Enhancement with Task-Oriented Flow},
  author    = {Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T.},
  booktitle = {IJCV},
  year      = {2019}
}

% === General Deep Learning ===

@inproceedings{vaswani_attention_2017,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle = {NeurIPS},
  year      = {2017}
}

@inproceedings{he_resnet_2016,
  title     = {Deep Residual Learning for Image Recognition},
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {CVPR},
  year      = {2016}
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd zjuthesis && latexmk
```

Fix any undefined citation warnings (these are expected until chapters reference them).

- [ ] **Step 3: Commit**

```bash
git add zjuthesis/body/ref.bib
git commit -m "refs: replace VOS references with VSR/diffusion/SSM bibliography"
```

---

### Task 5: Deep literature research (before any thesis writing) — NOT STARTED

**Target:** Apr 9–18, after meeting confirms research direction.

**Files:**
- Create: `docs/research/vsr-literature-notes.md`

This task MUST be completed before Tasks 6, 7 (Introduction, Lit Review). Writing without thorough research produces shallow content. The goal is to read and understand key papers, identify gaps, and collect precise claims/numbers to cite.

- [ ] **Step 1: Survey VSR papers — identify all relevant methods and their limitations**

Search for and read (at minimum) the following categories of papers. For each, note: key contribution, architecture, reported metrics, sequence length tested, limitations.

**CNN-based VSR:**
- BasicVSR (CVPR 2021), BasicVSR++ (CVPR 2022) — bidirectional recurrent propagation
- EDVR (CVPRW 2019) — deformable alignment
- RealBasicVSR (CVPR 2022) — real-world degradation pipeline
- IconVSR (CVPR 2021) — information-refill mechanism
- TOF / TOFlow (IJCV 2019) — task-oriented flow

**Transformer-based VSR:**
- VRT (TIP 2024) — mutual attention across frames
- RVRT (NeurIPS 2024) — recurrent + transformer
- PSRT (NeurIPS 2022) — patch alignment + recurrent transformer
- TTVSR (CVPR 2022) — trajectory-aware transformer

**Diffusion-based VSR:**
- Upscale-A-Video (CVPR 2024) — our baseline
- MGLD-VSR (ECCV 2024) — our baseline
- StableSR (IJCV 2024) — image SR with SD prior
- ResShift (NeurIPS 2023) — efficient diffusion for SR
- SUPIR (2024) — scaling up restoration with diffusion
- Any other 2024-2025 diffusion VSR papers found during search

**State-Space Models:**
- S4 (ICLR 2022) — structured state spaces
- Mamba (arXiv 2312.00752) — selective state spaces
- Mamba-2 (arXiv 2405.21060) — SSD duality
- Vision Mamba / VMamba (ICML 2024) — SSM for vision
- VMambaIR (arXiv 2403.11423) — SSM for image restoration
- Po et al. (arXiv 2505.20171) — long-context SSM video world models
- Any SSM + video generation/restoration papers from 2024-2025

**Long-video processing:**
- Search for papers that specifically address long video sequences (>100 frames) in any restoration task
- Note: this is likely sparse — the gap itself is a finding

- [ ] **Step 2: For each paper, extract key data points**

For every paper reviewed, record in `docs/research/vsr-literature-notes.md`:

```markdown
### [Paper Name] — [Venue Year]
- **Key idea:** [1-2 sentences]
- **Architecture:** [CNN/Transformer/Diffusion/SSM + key components]
- **Temporal modeling:** [how it handles frame-to-frame dependencies]
- **Max sequence length tested:** [frames]
- **Key metrics:** [best PSNR/SSIM on which dataset]
- **Limitations:** [what they don't solve]
- **Relevant to our thesis because:** [connection]
```

- [ ] **Step 3: Identify the research gap and frame the thesis contribution**

After surveying all papers, write a "Research Gap" section in the notes:
- What sequence lengths have been tested across all papers?
- Which methods claim temporal consistency but only test on <100 frames?
- Has anyone combined SSMs with diffusion for video SR?
- What specific claims can we make about the novelty of our approach?

- [ ] **Step 4: Collect precise numbers for comparison tables**

Build a comparison table with reported metrics from all relevant methods on common benchmarks (UDM10, REDS, Vid4, Vimeo-90K). These numbers will go directly into the thesis experiments chapter later.

- [ ] **Step 5: Verify and augment ref.bib entries**

Cross-check that all papers found have correct BibTeX entries. Search for the official BibTeX from DBLP, Semantic Scholar, or the paper's official page. Add any new references found during the survey that aren't already in the ref.bib draft from Task 4.

- [ ] **Step 6: Commit research notes**

```bash
mkdir -p docs/research
git add docs/research/vsr-literature-notes.md
git commit -m "research: deep literature survey — VSR, diffusion, SSM, long-video"
```

---

### Task 6: Rewrite Introduction chapter — NOT STARTED (blocked by Task 5)

**Files:**
- Modify: `zjuthesis/body/graduate-eng/introduction.tex`

The current file contains 182 lines of VOS content (reasoning segmentation, MLLMs, VOS, RVOS). Replace entirely with VSR content. Structure:

1. **Introduction** — Problem statement (video SR for long videos)
2. **Motivation** — Why long videos are hard (memory, temporal consistency)
3. **Contributions** — Preliminary list (will be refined after method is confirmed)
4. **Thesis Organization** — Chapter overview

- [ ] **Step 1: Write the Introduction section**

Replace the entire file with:

```latex
\chapter{Introduction}

\section{Background and Motivation}

Video super-resolution (VSR) aims to reconstruct high-resolution (HR) video sequences from their
low-resolution (LR) counterparts, and has become a fundamental task in computer vision with
applications ranging from surveillance enhancement to video streaming and film restoration.
Unlike single image super-resolution (SISR), VSR methods must exploit temporal correlations
across frames to produce spatially sharp and temporally consistent outputs.

The field has progressed through several paradigm shifts. Early approaches relied on optical
flow-based alignment followed by convolutional fusion~\cite{wang_edvr_2019, chan_basicvsr_2021}.
Transformer-based methods such as VRT~\cite{liang_vrt_2022} and RVRT~\cite{liang_rvrt_2024}
introduced attention mechanisms for implicit alignment, achieving state-of-the-art quality on
standard benchmarks. Most recently, diffusion-based approaches~\cite{zhou_upscale_a_video_2024,
yeung_mgldvsr_2024} have demonstrated superior perceptual quality by leveraging pretrained
generative priors from large-scale image diffusion models.

However, a critical limitation pervades existing methods: \textbf{they are designed for and
evaluated on short video clips}, typically 30--100 frames (1--4 seconds). Real-world video
content --- films, surveillance footage, user-generated content --- routinely spans minutes to
hours. When applied to long videos ($>$1000 frames), current methods face fundamental challenges:

\begin{itemize}
    \item \textbf{Memory scaling:} Attention-based methods exhibit $O(n^2)$ memory complexity
          with respect to sequence length, making them impractical for long sequences on
          consumer or even professional hardware.
    \item \textbf{Temporal consistency degradation:} Recurrent methods accumulate errors over
          time, leading to flickering artifacts and color drift in long sequences.
    \item \textbf{Computational cost:} Diffusion-based methods require multiple denoising steps
          per frame, and their temporal modules add further overhead that scales poorly with
          video length.
\end{itemize}

State-space models (SSMs)~\cite{gu_s4_2022, gu_mamba_2023} offer a promising alternative for
long-range temporal modeling. With $O(n)$ time and memory complexity and the ability to maintain
persistent state across arbitrarily long sequences, SSMs have shown strong results in natural
language processing and, more recently, in vision tasks~\cite{zhu_vmamba_2024, shi_vmambair_2024}.
Recent work on long-context video world models~\cite{po_longcontext_ssm_2025} has demonstrated
that SSMs can effectively model temporal dependencies over thousands of frames, suggesting their
potential for long-video super-resolution.

This thesis investigates the application of state-space models to enable temporally consistent,
memory-efficient video super-resolution for long videos exceeding one minute in duration.

\section{Research Objectives}

The primary objectives of this research are:

\begin{enumerate}
    \item To analyze the failure modes of existing VSR methods (both diffusion-based and
          attention-based) when applied to long video sequences, quantifying degradation in
          temporal consistency, memory usage, and output quality as sequence length increases.
    \item To design a VSR architecture that leverages state-space models for efficient
          long-range temporal feature propagation, enabling processing of videos with
          $>$1000 frames without quality degradation.
    \item To evaluate the proposed method on both standard short-video benchmarks (UDM10, REDS,
          Vid4) for comparison with existing work, and on a long-video benchmark for validating
          the core contribution.
\end{enumerate}

\section{Thesis Organization}

The remainder of this thesis is organized as follows:

\begin{itemize}
    \item \textbf{Chapter 2} reviews related work in video super-resolution, diffusion models,
          and state-space models, establishing the theoretical foundation for the proposed method.
    \item \textbf{Chapter 3} presents the proposed methodology, including the overall architecture,
          the state-space temporal module, and the training strategy.
    \item \textbf{Chapter 4} describes the experimental setup, presents quantitative and qualitative
          results on standard and long-video benchmarks, and provides ablation studies analyzing
          key design choices.
\end{itemize}
```

- [ ] **Step 2: Verify compilation**

```bash
cd zjuthesis && latexmk
```

Check for undefined citation warnings — all `\cite{}` keys should match `ref.bib`. Fix any compilation errors.

- [ ] **Step 3: Commit**

```bash
git add zjuthesis/body/graduate-eng/introduction.tex
git commit -m "thesis: rewrite Introduction chapter for VSR topic"
```

---

### Task 7: Rewrite Literature Review chapter — NOT STARTED (blocked by Task 5)

**Files:**
- Modify: `zjuthesis/body/graduate-eng/literature-review.tex`

The current file is a 35-line template placeholder. Replace with a structured review covering:

1. Video Super-Resolution (CNN → Transformer → Diffusion evolution)
2. Diffusion Models for Super-Resolution
3. State-Space Models
4. Long-Video Processing

- [ ] **Step 1: Write the Literature Review**

Replace the entire file with:

```latex
\chapter{Literature Review}

This chapter reviews the key research areas that form the foundation of this thesis:
video super-resolution methods, diffusion models for image and video restoration,
and state-space models for efficient sequence modeling.

\section{Video Super-Resolution}

Video super-resolution has evolved through three major paradigms: CNN-based methods with
explicit alignment, transformer-based methods with implicit alignment, and diffusion-based
methods with generative priors.

\subsection{CNN-Based Methods}

Early deep learning approaches to VSR focused on explicit motion compensation through optical
flow estimation followed by convolutional feature fusion. EDVR~\cite{wang_edvr_2019} introduced
deformable convolutions for alignment without explicit flow computation, using a pyramid-based
alignment module and temporal attention fusion. This approach demonstrated that learned alignment
could outperform optical flow-based warping for the super-resolution task.

BasicVSR~\cite{chan_basicvsr_2021} systematically analyzed the essential components of VSR
pipelines, finding that a bidirectional recurrent architecture with optical flow-based
propagation (using SpyNet~\cite{ranjan_spynet_2017}) achieves a strong balance of quality and
efficiency. Its successor, BasicVSR++~\cite{chan_basicvsrpp_2022}, added second-order grid
propagation and flow-guided deformable alignment, establishing new state-of-the-art results on
REDS~\cite{nah_reds_2019} and Vimeo-90K~\cite{xue_vimeo90k_2019}.

RealBasicVSR~\cite{chan_realbasicvsr_2022} addressed the domain gap between synthetic training
data and real-world degradations by introducing a practical two-stage degradation pipeline
(blur, noise, JPEG compression, video codec artifacts) that better simulates real-world
conditions. This degradation model has become the \textit{de facto} standard for evaluating
real-world VSR methods, including the baselines studied in this thesis.

The recurrent nature of CNN-based methods provides $O(n)$ memory complexity with respect to
sequence length, making them theoretically suitable for long videos. However, in practice,
error accumulation in the recurrent state leads to quality degradation over long sequences,
manifesting as temporal flickering and color drift.

\subsection{Transformer-Based Methods}

Transformer-based VSR methods replace explicit flow-based alignment with attention mechanisms
that implicitly model inter-frame correspondences.

VRT~\cite{liang_vrt_2022} applies mutual attention across frames, where queries from one frame
attend to keys and values from neighboring frames. While effective, VRT processes fixed-size
temporal windows, limiting its ability to capture long-range dependencies. RVRT~\cite{liang_rvrt_2024}
addresses this by combining recurrent processing with transformer attention, using guided
deformable attention to focus computation on relevant spatial locations.

PSRT~\cite{shi_rethinking_psrt_2022} rethinks alignment in VSR transformers, proposing
patch-level alignment that reduces computational cost while maintaining quality.

The fundamental limitation of transformer-based methods for long videos is the $O(n^2)$ memory
and computation cost of self-attention. While windowed attention and recurrent designs mitigate
this, they introduce compromises: windowed attention limits the receptive field, and recurrent
designs reintroduce the error accumulation problem of CNN-based methods.

\subsection{Diffusion-Based Methods}

Diffusion models~\cite{ho_ddpm_2020, rombach_ldm_2022} have recently been applied to video
super-resolution, leveraging the rich generative priors learned from large-scale image generation.

Upscale-A-Video~\cite{zhou_upscale_a_video_2024} (CVPR 2024) adapts the Stable Diffusion
framework for temporal-consistent video upscaling. It introduces a local-global temporal
strategy: local temporal layers model short-range frame-to-frame consistency, while a
flow-guided recurrent latent propagation module ensures long-range coherence. The method
also employs a text-guided latent refinement step using CLIP features to improve semantic
consistency. On standard benchmarks, Upscale-A-Video achieves strong perceptual quality
(PSNR 30.79 on UDM10, 25.83 on YouHQ40) while maintaining temporal smoothness.

MGLD-VSR~\cite{yeung_mgldvsr_2024} (ECCV 2024) proposes motion-guided latent diffusion for
real-world VSR. Its key innovation is a motion-guided sampling strategy that conditions the
diffusion process on estimated optical flow fields, encouraging the denoising trajectory to
respect inter-frame motion patterns. This explicit motion guidance improves temporal consistency
compared to methods that rely solely on temporal attention. MGLD-VSR also employs a
degradation-aware prompt extractor (DAPE) that adapts the diffusion process to the specific
degradation characteristics of the input.

Both methods demonstrate superior perceptual quality compared to regression-based approaches,
but share a common limitation: their temporal modules are designed for short clips and do not
scale efficiently to long sequences. Upscale-A-Video processes clips of 45 frames;
MGLD-VSR operates on 32-frame segments. Neither has been evaluated on videos exceeding
100 frames.

\section{Diffusion Models}

Denoising Diffusion Probabilistic Models (DDPMs)~\cite{ho_ddpm_2020} learn to reverse a
gradual noising process, generating samples by iteratively denoising from Gaussian noise.
The forward process adds noise over $T$ steps according to a fixed schedule:
\begin{equation}
    q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})
\end{equation}
where $\beta_t$ is the noise schedule. The reverse process is parameterized by a neural
network $\epsilon_\theta$ trained to predict the noise:
\begin{equation}
    p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})
\end{equation}

Latent Diffusion Models (LDMs)~\cite{rombach_ldm_2022} improve efficiency by operating in a
compressed latent space obtained through a pretrained autoencoder. This reduces computational
cost by a factor of $4\times$--$8\times$ while maintaining generation quality, and enables the
use of powerful pretrained models such as Stable Diffusion for downstream tasks including
super-resolution.

DDIM~\cite{song_ddim_2021} provides a deterministic sampling procedure that allows fewer
denoising steps while maintaining quality, which is particularly important for video applications
where per-frame inference cost directly impacts total processing time.

For super-resolution, diffusion models are typically conditioned on the low-resolution input,
either by concatenating it with the noisy latent or by injecting it through cross-attention.
StableSR~\cite{wang_stablesr_2024} demonstrated that pretrained Stable Diffusion provides
strong priors for image SR, achieving state-of-the-art perceptual quality.

\section{State-Space Models}

State-space models provide an alternative to attention for sequence modeling, offering
$O(n)$ time and memory complexity with respect to sequence length.

\subsection{Foundations}

The Structured State Space sequence model (S4)~\cite{gu_s4_2022} parameterizes a continuous-time
linear system:
\begin{equation}
    h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t), \quad y(t) = \mathbf{C}h(t) + \mathbf{D}x(t)
\end{equation}
where $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the state matrix, and $h(t)$ is the hidden
state. S4 initializes $\mathbf{A}$ with the HiPPO matrix, which provides optimal polynomial
approximation of input history, enabling long-range dependency modeling.

Mamba~\cite{gu_mamba_2023} extends S4 with input-dependent (selective) state transitions,
where the matrices $\mathbf{B}$, $\mathbf{C}$, and the discretization step $\Delta$ are
functions of the input. This selectivity allows the model to filter irrelevant information,
significantly improving performance on language and genomics tasks while maintaining linear
complexity. Mamba-2~\cite{dao_mamba2_2024} further refines this by establishing a formal
connection between SSMs and attention through structured state space duality (SSD), enabling
more efficient hardware implementations.

\subsection{SSMs in Vision}

Vision Mamba~\cite{zhu_vmamba_2024} adapts the Mamba architecture for visual representation
learning, introducing bidirectional scanning strategies to handle the 2D nature of images.
VMambaIR~\cite{shi_vmambair_2024} applies this to image restoration, demonstrating competitive
quality with transformer-based methods at lower computational cost.

For video, the key advantage of SSMs is their ability to maintain a compact state that
summarizes the entire processing history, without requiring attention over all past frames.
Po et al.~\cite{po_longcontext_ssm_2025} demonstrated this in a video world model context,
showing that SSMs can maintain spatial consistency over thousands of frames using a block-wise
scanning scheme that balances spatial coherence with temporal memory.

\subsection{Relevance to Long-Video Super-Resolution}

The properties of SSMs align directly with the requirements of long-video VSR:

\begin{itemize}
    \item \textbf{Linear complexity:} Processing cost scales linearly with video length,
          enabling practical inference on videos with thousands of frames.
    \item \textbf{Persistent memory:} The hidden state accumulates information from all
          previous frames without explicit storage, avoiding the memory overhead of
          attention-based approaches.
    \item \textbf{Selective processing:} Input-dependent state transitions (as in Mamba)
          can learn to retain temporally relevant features (consistent textures, recurring
          patterns) while discarding frame-specific noise.
\end{itemize}

To our knowledge, no existing work applies SSMs as the temporal backbone for diffusion-based
video super-resolution. This thesis explores this combination, hypothesizing that SSM-based
temporal propagation can maintain the perceptual quality of diffusion-based SR while scaling
efficiently to long videos.
```

- [ ] **Step 2: Verify compilation**

```bash
cd zjuthesis && latexmk
```

Check that all citations resolve and no compilation errors.

- [ ] **Step 3: Commit**

```bash
git add zjuthesis/body/graduate-eng/literature-review.tex
git commit -m "thesis: write Literature Review chapter (VSR, diffusion, SSM)"
```

---

### Task 8: Run baselines on long videos — document failure modes — NOT STARTED (blocked by sample data from Apr 9 meeting)

**Files:**
- Create: `experiments/baselines/long_video_eval/README.md`
- Modify: `experiments/baselines/target_metrics.md`

This task requires sample long-video data from the PhD student (expected at April 9 meeting). If data arrives earlier, proceed. Otherwise, this task starts after the meeting.

**Fallback if no long-video data by April 14:** Stitch UDM10 clips end-to-end to create a synthetic long sequence (>500 frames), or use the longest available clips from existing datasets.

- [ ] **Step 1: Create long-video eval directory and document methodology**

Create `experiments/baselines/long_video_eval/README.md`:

```markdown
# Long-Video Baseline Evaluation

## Purpose
Document how existing diffusion-based VSR baselines (Upscale-A-Video, MGLD-VSR) perform
on long videos (>500 frames / >20 seconds). This provides motivation for the thesis:
existing methods degrade on long sequences.

## What We Measure
1. **Can it run?** — Does the model OOM or crash on long sequences?
2. **Temporal consistency** — Does flickering/drift increase with sequence length?
3. **Metrics vs sequence position** — Plot PSNR/SSIM per frame to show degradation over time
4. **Memory usage** — Peak VRAM as a function of input length
5. **Processing time** — Per-frame time as input length increases

## Test Sequences
- Source: [TBD — PhD student sample data or stitched UDM10]
- Lengths tested: 100, 250, 500, 1000 frames
- Degradation: RealBasicVSR pipeline (same as short-video experiments)

## Results
[To be filled after experiments]
```

- [ ] **Step 2: Run MGLD-VSR on long sequence (server-side)**

MGLD-VSR processes 32 frames at a time with overlap. Test with increasing sequence lengths.

```bash
ssh -i ~/.ssh/id_ed25519_timuj Timur@223.109.239.43 << 'REMOTE'
tmux new-session -d -s mgld_long "
eval \"\$(/data/disk1/timur/miniconda3/bin/conda shell.bash hook)\"
conda activate mgldvsr
cd /data/disk1/timur/thesis_ve/experiments/baselines/mgld_vsr/repo

# Test with long sequence — use longest available clip or stitched sequence
# Record: peak VRAM (nvidia-smi), total time, output quality
nvidia-smi --query-gpu=memory.used --format=csv -l 5 > /data/disk1/timur/mgld_long_vram.log &
VRAM_PID=\$!

python scripts/vsr_val_ddpm_text_T_vqganfin_w_latent.py \
    --config configs/mgldvsr/mgldvsr_512_realbasicvsr_deg.yaml \
    --ckpt checkpoints/mgldvsr_unet.ckpt \
    --vqgan_ckpt checkpoints/video_vae_cfw.ckpt \
    --seqs-path /data/disk1/timur/data/LONG_VIDEO_LQ \
    --outdir /data/disk1/timur/thesis_ve/experiments/baselines/results/mgld_vsr/long_video \
    --latent-dir /tmp/mgld_latent_long \
    --ddpm_steps 50 --dec_w 1.0 --colorfix_type adain --n_gpus 1 \
    2>&1 | tee /data/disk1/timur/mgld_long_inference.log

kill \$VRAM_PID
echo 'DONE'
"
REMOTE
```

Note: Replace `/data/disk1/timur/data/LONG_VIDEO_LQ` with actual path once data is available.

- [ ] **Step 3: Run UAV on long sequence (server-side)**

Same approach for Upscale-A-Video. UAV processes in chunks and may hit issues with long sequences.

- [ ] **Step 4: Analyze results — per-frame metrics and VRAM profile**

```bash
# On server: evaluate per-frame metrics (not just average)
/data/disk1/timur/miniconda3/envs/vsr/bin/python experiments/baselines/evaluate_pyiqa.py \
    --sr results/mgld_vsr/long_video/<clip> \
    --gt /data/disk1/timur/data/LONG_VIDEO_GT/<clip> \
    --output results/mgld_vsr/long_video_metrics.json \
    --per_frame  # if supported, or modify script to output per-frame
```

- [ ] **Step 5: Document findings in README.md and commit**

Update `experiments/baselines/long_video_eval/README.md` with actual results. Key findings to capture:
- At what sequence length does each model start to degrade?
- What's the failure mode (OOM, temporal drift, quality loss)?
- VRAM profile chart data

```bash
git add experiments/baselines/long_video_eval/
git commit -m "experiments: document baseline failure modes on long videos"
```

---

### Task 9: Prepare baseline presentation for April 9 meeting — DONE

**Files:**
- Already exists: `reports/presentation_baseline_methods.md`

18-slide presentation completed last week. Should be updated with:
- [x] MGLD-VSR verified results
- [x] UAV results (RealBasicVSR gap + DOVE cross-validation)
- [x] Key finding: degradation mismatch explains the gap
- [ ] **Step 1: Add UAV RealBasicVSR gap + param sweep findings to presentation before meeting**

---

## Dependency Graph (updated April 7)

```
Task 1 (UAV UDM10 eval)    ── DONE
Task 2 (UAV YouHQ40 eval)  ── DONE
Task 9 (presentation)      ── DONE (update with gap findings before Apr 9)

Task 3 (thesis metadata)   ── deferred until direction confirmed (Apr 9)
Task 4 (ref.bib)           ── REMOVED (incremental approach)

Task 5 (deep research)     ── NOT STARTED → start Apr 9 ──→ Task 6 (Introduction)
                                                          └──→ Task 7 (Lit Review)
                                                          └──→ ref.bib entries added as needed

Task 8 (long-video eval)   ── blocked by sample data (April 9 meeting)

NEW: UAV VideoLQ NR eval   ── RUNNING (2/50 clips, ~4 days)
NEW: Param sweep           ── DONE (5 configs, ruled out hyperparameters)
NEW: DOVE cross-validation ── DONE (pipeline verified)
```

**Critical path:** April 9 meeting → Task 5 (literature review) → Tasks 6-7 (writing). Must start Task 5 immediately after meeting.
Task 8 is blocked until long-video data is available from PhD student.
UAV VideoLQ NR eval runs autonomously on server.
