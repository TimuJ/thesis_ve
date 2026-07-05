# SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training

**arXiv 2506.05301v2 (v2: 28 Jan 2026) — Wang et al., NTU (MMLab) + ByteDance Seed. Project: https://iceclear.github.io/projects/seedvr2/**

Q&A prep notes for the presentation (`../seedvr2_flashvsr_slides.pptx`).

---

## 1. One-sentence pitch

SeedVR2 turns the multi-step diffusion-transformer video restorer SeedVR into a **single-step** generator via GAN-style **Adversarial Post-Training (APT) against real data** (no frozen teacher/prior at the adversarial stage), enabled by an **adaptive window attention** for arbitrary/high resolutions and a set of stabilizing losses (RpGAN + approximated R1/R2 + a discriminator **feature matching loss** replacing LPIPS) — yielding the largest VR GAN to date (~16B generator + discriminator combined) that matches or beats 50-step diffusion VR baselines at **>4× lower runtime**.

## 2. Problem & motivation

- **Diffusion VR is slow.** Diffusion-based real-world video restoration (MGLD-VSR, UAV, VEnhancer, STAR, SeedVR) gives realistic textures but needs tens of sampling steps; the cost is "further amplified" for long, high-resolution videos.
- **One-step distillation methods don't transfer well to VR.** Existing one-step IR approaches mostly distill from a multi-step teacher → (a) student quality is upper-bounded by the teacher, (b) pre-computing teacher samples is expensive, (c) image-only methods lack temporal design. Distillation-free one-step methods (discriminator prior, generative prior + LoRA) still lean on frozen priors that carry their own biases.
- **SeedVR v1's specific limitations addressed here:** (i) still requires tens of steps; (ii) its shifted window attention uses a **predefined window size** — at >2K test resolutions this produces visible **boundary artifacts** between window patches and weakens robustness of the 3D RoPE inside windows for unseen resolutions; (iii) naive adversarial training of a model this large deteriorates after long training (~20k iterations) without extra stabilization.
- Positioning claim: "among the early attempts" at one-step diffusion-transformer video restoration; adversarial training **against real data without a frozen prior** lets the student potentially *surpass* its initialization — impossible under teacher distillation.

## 3. Method

### 3.1 Architecture (Fig. 2, p.5)

- **Generator:** SeedVR's diffusion transformer — a **Swin-MMDiT** with **36 blocks**, now with **adaptive window attention** replacing fixed windows. Inputs: noise + noised LQ video (latent) + text condition + timestep T. Trained on the **velocity field** (v-prediction) with flow matching; linear noise schedule, timesteps 0–999.
- **VAE:** SeedVR's **causal video VAE** (temporally causal, from v1; downsampling factor 8 spatially). Kept as-is — and flagged as the main efficiency bottleneck (see Limitations).
- **Discriminator:** initialized from the **same pre-trained diffusion transformer** (APT recipe). Additional **cross-attention-only transformer blocks + MLPs** are inserted, tapping features after **blocks 16, 26, and 36** of the backbone; outputs are combined (LayerNorm + Linear) into logits for the GAN loss. Discriminator input: v→x converted prediction + noised LQ video + text. Generator + discriminator together ≈ **16B params** ("largest-ever VR GAN").
- **Sizes:** **Ours-7B** (8239.6M generator params, same count as SeedVR-7B) and **Ours-3B** (3391.5M), the 3B distilled from the 7B.

### 3.2 Adaptive window attention (the architectural contribution)

**Training-time (Eq. 1):** given a video feature $X \in \mathbb{R}^{d_t \times d_h \times d_w \times d_c}$, window sizes are computed as
$$p_t = \lceil \min(d_t, 30)/n_t \rceil,\quad p_h = \lceil d_h/n_h \rceil,\quad p_w = \lceil d_w/n_w \rceil$$
i.e., the spatial partition is fixed at **3×3 windows over the (resized) input** rather than a fixed pixel window size; $d_h \times d_w = 45 \times 80$ is the feature resolution at 720p, and the min(·,30) caps temporal window length to bound the train/test sequence-length gap. Because training clips are ~720p but with **widely varying aspect ratios**, the model naturally sees many different window sizes during training → generalization to diverse window shapes.

**Test-time resolution-consistent windowing (Eq. 2):** for a test feature $\hat{X}$ of resolution $\hat{d}_h \times \hat{d}_w$, compute a **proxy resolution** preserving the test aspect ratio but matching training area:
$$\tilde{d}_h = \sqrt{d_h d_w \cdot \hat{d}_h/\hat{d}_w},\qquad \tilde{d}_w = \sqrt{d_h d_w \cdot \hat{d}_w/\hat{d}_h}$$
so $\tilde{d}_h \tilde{d}_w = d_h d_w$ (= 45×80). Plug $(\hat{d}_t, \tilde{d}_h, \tilde{d}_w)$ into Eq. 1 to get the window size, then partition the actual high-res feature with windows of that size (chunking at borders). Net effect: **window sizes at test time match the statistics seen in training**, regardless of output resolution (e.g., 1080p, 2K) → eliminates the boundary artifacts of fixed 64×64-latent windows. Diagnosis in the ablation: a 64×64 window on 8×-downsampled latents means 720p training pairs almost never produce window-overlap/shift cases, so the model is undertrained on them; RoPE also generalizes better when window sizes vary in training.

### 3.3 Training pipeline: progressive distillation → adversarial post-training

**Stage 0 — retrain SeedVR-7B from scratch** with the new adaptive window attention (each training stage ≈ 1 day on the cluster).

**Stage 1 — progressive distillation (Salimans & Ho 2022):** start from the teacher at **64 sampling steps**, halve steps repeatedly (**stride 2**: 64→32→…→1), each halving trained ~**10K iterations** with a **simple MSE loss on the velocity field** (flow matching). Motivation: jumping straight from a multi-step model to one-step adversarial training loses restoration ability under **heavy degradations** (the paper notes VR is otherwise more stable than T2V APT — no mode collapse even with a single adversarial stage, thanks to the LQ conditioning). During adversarial training they also **progressively increase temporal length** of training data from images up to variable-length clips → robustness across video lengths including single images. The **3B model** is obtained by distilling from the 7B, with comparable performance at half the size.

**Stage 2 — adversarial post-training against real data** (following APT, Lin et al. 2025, but with modified losses):

- **GAN loss:** replace APT's non-saturating GAN loss with **RpGAN (relativistic pairing GAN, Jolicoeur-Martineau 2019)**, following R3GAN ("The GAN is dead…", Huang et al. 2024), to avoid mode dropping.
- **Approximated R1** (from APT, approximated to avoid higher-order gradients) **+ new approximated R2** on fake data (Eq. 3):
$$\mathcal{L}_{aR2} = \| D(\hat{x}, c) - D(\mathcal{N}(\hat{x}, \sigma \mathbf{I}), c) \|_2^2$$
where $\hat{x}$ is the sample prediction converted from the velocity output, $c$ the text condition — i.e., penalize the discriminator's sensitivity to small Gaussian perturbations of fake samples (finite-difference proxy for the gradient-norm penalty).
- **Feature matching loss** replacing LPIPS (Eq. 4):
$$\mathcal{L}_F = \frac{1}{3}\sum_{i \in \{16,26,36\}} \| D_i^F(\hat{x}, c) - D_i^F(x, c) \|_1$$
L1 distance between discriminator features of prediction vs. ground truth, extracted **before the cross-attention-only blocks** at backbone blocks 16/26/36. Rationale: LPIPS requires decoding latents to pixel space every step — unaffordable for HR video — and no latent-LPIPS exists for video; here the (frozen-during-G-updates) discriminator plays the role VGG plays in LPIPS.
- **Loss weights:** generator update — 1.0 each for L1, feature matching, GAN by default; discriminator update — 1.0 GAN, **1000** each for approx. R1 and R2. For the **final model**, L1 and feature-matching weights are reduced to **0.1** (large weights improve fidelity metrics but over-smooth — perception–distortion tradeoff).

### 3.4 Training data & recipe highlights

- **72 NVIDIA H100-80G**, ~100 frames of 720p per batch, sequence parallel + data parallel; AdamW, wd 0.01, LR **1e-6**; PyTorch 2.4 / CUDA 12.4.
- Data synthesized following **UAV** (Upscale-A-Video): ~**10M image pairs + 5M video pairs**.
- Teacher uses Euler sampler, **CFG 7.5 for 64-step stage, 1.0 afterwards**; distillation loss on the vector field (flow matching).

## 4. Key results

**Benchmarks:** synthetic — SPMCS, UDM10, REDS30, YouHQ40 (720p output, ×4 upscaling, UAV degradations); real — VideoLQ; AIGC — self-collected AIGC28 (28 AI-generated videos). Full-reference: PSNR/SSIM/LPIPS/DISTS; no-reference: NIQE/MUSIQ/CLIP-IQA/DOVER (+ warping error $E_{warp}^*$ and VMAF in appendix).

**Headline quantitative (Table 1, p.7; 1 step vs. 50 steps for diffusion baselines):**
- **Perceptual metrics on synthetic sets — best or 2nd-best almost everywhere.** E.g., UDM10: Ours-7B LPIPS **0.203** / DISTS **0.101** / SSIM **0.798** / PSNR 26.26 (vs. SeedVR-7B-50step: LPIPS 0.264, DISTS 0.124); SPMCS: Ours-3B LPIPS **0.306**, DISTS **0.131**; YouHQ40: Ours-7B LPIPS **0.274**, DISTS **0.110**. PSNR leaders are RealViformer/MGLD-VSR, but the paper notes REDS30 leaders trained on REDS.
- **Real/AIGC no-reference:** AIGC28 — Ours-3B best NIQE **3.801**, MUSIQ **62.99**, DOVER **15.77** (7B: 15.55). VideoLQ — Ours-3B best DOVER **8.176**, NIQE 4.687 (authors caution NR metrics prefer over-sharpening).
- **Speed (Table 6, p.19; 100 frames, 768×1344):** Ours-3B **269.0 s**, Ours-7B **299.4 s** vs. MGLD-VSR 1181, UAV 1284.5, SeedVR-7B(50-step) 1284.8, VEnhancer 2029.2, STAR 2326 → **>4× faster** despite ≥4× more parameters than most baselines.
- **User study (GSB, Table 2, p.8):** vs. Ours-7B as datum: SeedVR-7B-50 +10% overall, Ours-3B-1 **+16%** overall, all other baselines strongly negative (VEnhancer −94%). Humans preferred the 3B over the 7B, credited to the distillation stage.
- **Temporal:** $E_{warp}^*$ is won by bicubic (authors argue warping error is a flawed metric for generative VR); SeedVR2 leads **VMAF** on 3/4 synthetic sets and DOVER; temporal-profile visualization (Fig. 6) shows clean temporal edges.

**Ablations (Table 3, p.9; YouHQ40, 20k iters, 72×H100):**
- Losses: Non-saturating+R1 (APT default) → RpGAN+R1+R2 → +L1 → +feature-matching: LPIPS 0.310 → 0.278 → 0.251 → **0.244**; DISTS 0.136 → 0.109 → 0.099 → **0.092**. RpGAN+R2 is the biggest single jump and prevents late-training collapse; L1 + feature matching matter for restoration fidelity.
- **Progressive training** column: PSNR 23.96, SSIM 0.667, LPIPS 0.227 — needed to preserve restoration strength under heavy degradations.
- **Adaptive vs. predefined window attention (Fig. 4, p.9):** predefined windows give visible seam/boundary artifacts at 1080p; adaptive removes them.
- **Adversarial vs. pure progressive distillation (Table 5, p.18):** on VideoLQ, Ours-3B beats the distillation-only baseline (NIQE 4.687 vs 5.365, MUSIQ 51.09 vs 45.57, DOVER 8.176 vs 6.609) at equal iterations → evidence for surpassing-the-teacher claim.
- **Speed–quality tradeoff (Fig. 10, p.22):** multi-step baselines degrade sharply on perceptual metrics below ~25 steps — one-step is not achievable by simply reducing their steps.
- **Concurrent work:** vs. DOVE-5B (Table 7, p.21) — Ours-3B (1.67× smaller) better NIQE/MUSIQ, slightly worse CLIP-IQA/DOVER; vs. DLoRAL (Fig. 11) — better textures, less flicker.

## 5. Limitations (authors' own, Sec. 4.3, p.10)

1. **Causal video VAE is the bottleneck:** >4× slower to encode/decode than the VAEs used by baselines; for a 100-frame 720p video the VAE takes **>95% of total runtime** (so the DiT itself is far faster than the headline 4× suggests).
2. **Not robust to heavy degradations / very large motions** — shares failure cases with existing methods.
3. **Over-sharpening on lightly degraded inputs** (e.g., 720p AIGC video); guaranteeing a performance lower bound remains open.

## 6. Figure inventory (used in slides)

| Item | PDF page | What it shows |
|---|---|---|
| Figure 1 | p.2 | Hero: speed vs. LPIPS bubble chart + qualitative strip |
| Figure 2 | p.5 | Architecture: generator/discriminator Swin-MMDiT, taps at blocks 16/26/36 |
| Table 1 | p.7 | Main quantitative comparison, 6 benchmarks × 8 methods |
| Table 3 + Fig. 4 | p.9 | Loss ablation; window-attention boundary artifacts |
| Table 6 + Fig. 10 | p.19/22 | Inference time; speed–quality tradeoff curves |

## 7. Relevance hooks for long-video SR (thesis context)

- **Cost scaling:** one-step inference is a ~50× reduction in DiT passes, orthogonal to sequence-length methods (SSM backbones, streaming). But with the DiT at 1 step, the **causal video VAE dominates (>95%) runtime** — for long videos, VAE efficiency becomes the wall (explicitly stated open direction).
- **Window attention + RoPE robustness:** fixed window sizes break RoPE generalization at unseen resolutions; training with *varying* window sizes makes RoPE robust — directly relevant to our RoPE probe / position-interpolation work on FlashVSR (same failure mode, spatial instead of temporal). Their Eq. 1 caps the temporal window at **30 latent frames** — they sidestep temporal extrapolation rather than solve it.
- **Temporal consistency evaluation:** warping error is unreliable for generative VR (bicubic wins it); they lean on VMAF/DOVER + temporal profiles + user study — useful for our own long-video evaluation protocol choices.
- **Attention is still windowed/local in time** (≤30-frame temporal windows with shifting); nothing provides global long-video memory — SSM/state-space cross-window temporal state remains complementary and unexplored.
- **Training-length curriculum:** progressively growing clip length during adversarial training is a transferable trick for length robustness.

**Ambiguities to flag if asked:** exact $n_t, n_h, n_w$ values only implied (3×3 spatial; $n_t$ not stated); "~16B GAN" is generator+discriminator combined; VAE architecture details inherited from SeedVR, not restated; per-stage "~1 day" on 72×H100 but total wall-clock not summed; Table 6 timing includes VAE.
