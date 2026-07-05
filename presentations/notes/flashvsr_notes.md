# FlashVSR: Towards Real-time Diffusion-Based Streaming Video Super-Resolution

**arXiv 2510.12747v1 — Zhuang, Guo, Cai, Li, Liu, Yuan, Xue (Tsinghua / Shanghai AI Lab / CUHK / SJTU), Oct 2025**

Q&A prep notes for the presentation (`../seedvr2_flashvsr_slides.pptx`).

---

## 1. One-sentence pitch

FlashVSR is the first one-step, causally-streaming diffusion VSR framework — a Wan2.1-1.3B DiT distilled via a three-stage pipeline with block-sparse locality-constrained attention and a tiny LR-conditioned decoder — reaching ~17 FPS at 768×1408 on a single A100 (11.8× faster than the fastest prior one-step diffusion VSR, SeedVR2-3B) while scaling reliably to 1440p+.

## 2. Problem & motivation

Three obstacles blocking practical diffusion-based VSR:

1. **High lookahead latency of chunk-wise processing.** Prior methods split long videos into overlapping chunks; a frame is not output until its whole chunk (~80–101 frames) finishes. FlashVSR's streaming design has only **8 frames** of lookahead latency (vs. 32 for STAR, 101 for whole-sequence methods).
2. **Quadratic cost of dense 3D attention** — prohibitive for long, high-resolution videos.
3. **Train–test resolution gap.** VSR models trained at medium resolution degrade at ultra-high resolution (1440p+) with repeated patterns/blur; traced to **mismatched RoPE ranges between training and inference**.

Secondary motivation: existing autoregressive streaming diffusion uses teacher forcing (train–infer gap, error accumulation) or self forcing (consistent but **serial** training). FlashVSR's key insight: **VSR is strongly conditioned on LR frames, so clean historical latents are unnecessary for motion plausibility** — enabling fully parallel training with no train–infer gap.

## 3. Method

### 3.1 Overall architecture & base model
- Backbone: **Wan 2.1–1.3B** video DiT, fine-tuned with **LoRA (rank 384)**.
- One-step generator `G_one`: for frame block t, `z_t = G_one(LR_t, ε_t; KV_<t)` — current LR frames + Gaussian noise + sliding-window KV-cache of past keys/values. No past predicted latents fed back.
- **Causal LR Projection-In layer** (App. B.1, Fig. 6, p.17): replaces VAE-encoding of LR input. Every 4 LR frames → 2D pixel-shuffle (8× spatial, matching Wan's VAE) → two 3D CausalConv layers (each 2× temporal) → MLP → embedding **added element-wise to patchified latent tokens**. A causal feature cache carries the previous clip's representation forward. Negligible overhead.
- **Fixed text prompt** for all content, so cross-attention K/V are computed once and shared — no captioning cost; enables joint image–video training.

### 3.2 Three-stage distillation pipeline (Fig. 2, p.5)
- **Stage 1 — Video–Image Joint SR Training (teacher).** Full-attention DiT on videos + images (images = single-frame videos, unified 3D attention). **Block-diagonal segment mask** (Eq. 1) so images and video clips pack together. Loss: **flow matching**. ~2 days, 89-frame clips at 768×1280 + paired images.
- **Stage 2 — Block-Sparse Causal Attention Adaptation.** Convert to sparse-causal DiT: causal masking + block-sparse attention. LR Proj-In made causal. Flow matching, video data only. ~1 day.
- **Stage 3 — Distribution-Matching One-Step Distillation (DMD, Yin et al. 2024a).** All latents at a **unified timestep** with the block-sparse causal mask — training **parallel over frames** (no serial unfolding, unlike Self-Forcing/AAPT). Stage-1 full-attention DiT = frozen real-score teacher `G_real`; trainable copy `G_fake` tracks the fake distribution. Objective (Eq. 2):
  `L = L_DMD(z_pred, G_one, G_real, G_fake) + L_FM(z_pred, G_fake) + ||x_pred − x_gt||²₂ + λ·L_LPIPS(x_pred, x_gt)`, **λ = 2**.
  Only **2 latents per iteration are randomly decoded** for the pixel losses (memory), earlier ones detached. ~2 days.
- Infra: 32× A100-80G, batch 32, AdamW, lr 1e-5, wd 0.01. LR–HR pairs via **RealBasicVSR degradation pipeline**.

### 3.3 Block-sparse attention
- Q/K partitioned into latent blocks of size **(2, 8, 8)** (t, h, w) → 128-token blocks.
- **Average pooling per block** → coarse block-to-block attention → **top-k** block pairs → full **128×128** attention only there.
- Cuts attention cost to **10–20% of dense**; ablation at **13.6% sparsity**. Train–inference consistency: Stages 2–3 train through the same sparse mask used at inference (claimed first diffusion VSR with sparse attention).

### 3.4 Locality-constrained attention (resolution-scaling fix; Fig. 3, p.6)
- Diagnosis: RoPE is periodic; inference positions beyond the trained range make some dimensions repeat → repeated textures / blur at 1440p+.
- Fix: restrict each query to a **limited spatial neighborhood** so *relative* positional range matches training. Two window rules: **Boundary-Preserved** and **Boundary-Truncated** (differ near frame edges). Final mask = local ∩ top-k block-sparse. Ablation receptive field **1152×1152**. Training-free at deployment.

### 3.5 Tiny Conditional (TC) Decoder (Fig. 4, p.6)
- After one-step distillation, the causal 3D VAE decoder is **~70% of inference time** at 768×1408.
- TC Decoder conditions on **latents + pixel-shuffled LR frames** (LR carries low-frequency structure; decoder only reconstructs details).
- Trained separately (~2 days, 61-frame 384×384 clips), pixel supervision + distillation from Wan decoder (Eq. 3): `L = ||x_pred − x_gt||² + λ·LPIPS(x_pred, x_gt) + ||x_pred − x_wan||² + λ·LPIPS(x_pred, x_wan)`, λ = 2.
- **~7× faster decoding** (1.60 s vs. 11.13 s, 101 frames @ 768×1408); at equal params beats unconditional tiny decoder (Table 4: PSNR 31.08 vs. 29.96; Wan reference 32.58).

### 3.6 KV-cache streaming & temporal consistency
- Sliding-window KV-cache over most recent latents, all layers; ablation window = **85 frames**. Constant-memory streaming.
- **Why consistency holds without feeding back predictions** (App. B.4): early-layer KV aggregates LR-aligned structural/motion cues; later-layer KV holds progressively cleaner latents that stabilize textures across frames. Temporal alignment is implicit in the cache, not autoregressive conditioning.
- **Eviction ablation** (Table 6, p.17): sliding-window ≈ uniform importance-based eviction; **head-wise eviction clearly degrades** — *sink-attention*: some heads focus on the first frame, head-wise scoring keeps stale first-frame KVs → blur/distortion (Fig. 7).
- Fig. 8 (p.18): Teacher Forcing / AAPT / Self-Forcing / FlashVSR — FlashVSR is the only one with **both** parallel training and train–infer consistency.

### 3.7 VSR-120K dataset (Appendix A)
- Sources: Videvo, Pexels, Pixabay. Start: ~600k clips (>1080p) + 220k photos (shorter side >1024).
- Filtering: **LAION-Aesthetic + MUSIQ** (segment-level); **RAFT optical flow** to drop static clips.
- Final: **120k videos (avg >350 frames) + 180k images**; supports joint image–video training. (Contrast: DOVE dataset ≈ 2K videos.) To be released.

## 4. Key results

**Setup:** synthetic — YouHQ40, REDS, SPMCS (RealBasicVSR degradations); real — VideoLQ; AIGC30. Baselines: Upscale-A-Video (30 steps), STAR (15 steps), RealViformer (non-diffusion), DOVE, SeedVR2-3B (one-step). "Ours-Full" = Wan decoder, "Ours-Tiny" = TC decoder.

**Quality (Table 1, p.8)** — wins nearly all perceptual/no-reference metrics; PSNR/SSIM competitive, not best:
- YouHQ40: Ours-Full MUSIQ **69.16**, DOVER **12.71** (both best); LPIPS 0.3874 (2nd; SeedVR2 0.3876). Ours-Tiny NIQE 3.489.
- REDS: Ours-Full NIQE **2.425**, MUSIQ **68.97**, CLIPIQA **0.4661**, DOVER 8.734 (2nd to DOVE 9.368). Caveat: **RealViformer's REDS numbers inflated — REDS in its training set** (PSNR 25.96 tops table).
- SPMCS: NIQE **3.151**, MUSIQ **71.05**, CLIPIQA **0.5792**, DOVER 9.456 (2nd).
- VideoLQ: MUSIQ 55.48 (2nd to RealViformer 57.60), CLIPIQA **0.4184**, DOVER **8.149**.
- AIGC30: NIQE **3.871**, MUSIQ **56.89**, DOVER **12.65**.

**Efficiency (Table 2, p.8; 101 frames @ 768×1408, single A100):**

| | Upscale-A-Video | STAR | DOVE | SeedVR2-3B | Ours-Full | Ours-Tiny |
|---|---|---|---|---|---|---|
| Peak mem (GB) | 18.39 | 24.86 | 25.44 | 52.88 | 18.33 | **11.13** |
| Runtime (s) / FPS | 811.7 / 0.12 | 682.5 / 0.15 | 72.8 / 1.39 | 70.6 / 1.43 | 15.5 / 6.52 | **5.97 / 16.92** |
| Params (M) | 1086.75 | 2492.90 | 10548.57 | 3391.48 | 1780.14 | 1752.18 |

→ **136× faster than Upscale-A-Video, 114× than STAR, 11.8× than SeedVR2-3B**, ~4.8× less peak memory than SeedVR2. Lookahead: **8 frames** vs. 32 (STAR) / 101 (whole-sequence).

**Ablations:**
- **Sparse vs. full attention** (Table 3, REDS, KV-cache 85): 13.6% sparsity → PSNR 24.11 vs. 24.65 full, MUSIQ *better* (67.43 vs. 65.77); attention per 8 frames 1.105 s → 0.355 s (**3.1×**).
- **Locality-constrained attention** (Table 5, p.9; 15 videos at 1536×2688, avg 305 frames): both variants beat global on **all** metrics. Boundary-Preserved best fidelity (PSNR **24.87**, SSIM **0.7232**, LPIPS **0.3304**); Boundary-Truncated best perceptual (NIQE **2.850**, MUSIQ **67.47**, DOVER **9.5132**); Global: PSNR 24.21, DOVER 9.1259.
- **TC Decoder** (Table 4): Wan 32.58/0.9417/0.0715 (PSNR/SSIM/LPIPS) vs. TC **31.08/0.9244/0.1014** vs. unconditional tiny 29.96/0.9079/0.1231 — LR conditioning makes a tiny decoder viable; ~7× speedup.
- **KV eviction** (Table 6): sliding window is the pragmatic default; head-wise hurts (sink attention).
- **User study** (Table 7, p.22; 20 CV researchers, 32 test sets, GSB vs. Ours-Tiny): Ours-Full +2.2%, SeedVR2-3B −33.1%, DOVE −30.2%, RealViformer −44.9%.

## 5. Limitations

**No explicit "Limitations" section in the paper** — worth stating if asked. In-text caveats:
- TC Decoder trades fidelity for speed (−1.5 dB reconstruction PSNR).
- Sparse attention costs ~0.5 dB PSNR vs. full (perceptual metrics unaffected).
- PSNR/SSIM generally not SOTA — optimizes perceptual/no-reference quality.
- Importance-based KV eviction failed; only sliding-window works — smarter long-horizon memory open.
- Implicit: 4× SR with fixed synthetic degradation (RealBasicVSR) for synthetic evals; "near real-time" = 17 FPS on an A100; no explicit temporal-consistency metric (e.g., tOF); 8-frame lookahead small but nonzero.

## 6. Figure inventory (used in slides)

| Item | PDF page | What it shows |
|---|---|---|
| Fig. 1 | p.2 | Hero: MUSIQ vs. FPS bubble chart + visual comparison |
| Fig. 2 | p.5 | Three-stage training pipeline |
| Fig. 3 | p.6 | Locality-constrained attention / RoPE out-of-range artifacts |
| Tables 1 & 2 | p.8 | Quality across 5 benchmarks + efficiency table |
| Fig. 8 | p.18 | Four streaming training pipelines compared |

## 7. Relevance hooks for long-video SR (thesis context)

- **Constant-memory streaming:** sliding-window KV-cache + causal LR Proj-In cache → memory does not grow with length — directly applicable to >1-minute videos. But longest reported sequences ~305 frames average; behavior over thousands of frames is *not* demonstrated — a gap our work could probe.
- **Temporal consistency mechanism is the KV-cache, not recurrence:** early-layer KV = LR-aligned structure, late-layer KV = clean detail (App. B.4) is a testable hypothesis; sink-attention finding + eviction failures suggest long-horizon cache management is unsolved — connects to SSM-style constant-state alternatives (Po et al. long-context world models).
- **RoPE range mismatch as the cause of resolution-scaling failure** is directly adjacent to our rope-probe work: FlashVSR fixes the *spatial* PE mismatch via local windows; the analogous *temporal* PE extrapolation question is untouched.
- **Parallel-trainable streaming distillation** (no teacher/self forcing, because LR supplies motion) is VSR-specific — worth discussing when it breaks (heavy degradation, large motion where LR is uninformative).
- Establishes the current one-step diffusion VSR speed/quality frontier (17 FPS @ 768×1408, A100); releases code, weights, VSR-120K.
