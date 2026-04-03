# Diffusion-Based Real-World Video Super-Resolution
## Baseline Methods Report

---

## Slide 1: Overview

**Two state-of-the-art diffusion-based VSR methods:**

1. **Upscale-A-Video** (CVPR 2024) — Text-guided latent diffusion with local-global temporal strategy
2. **MGLD-VSR** (ECCV 2024) — Motion-guided latent diffusion with temporal-aware decoder

**Common challenge:** Diffusion models produce high-quality details but introduce temporal inconsistencies (flickering) in video — each frame is denoised independently with random noise.

---

## Slide 2: Problem Statement

**Video Super-Resolution (VSR):** Reconstruct HR video from LR input

**Real-world VSR** is harder than classical VSR:
- Unknown, complex degradations (blur + noise + compression + downsampling)
- Models trained on simple bicubic degradation fail on real videos
- Need to generate realistic details while maintaining temporal consistency

**Why diffusion?**
- CNNs (BasicVSR++, RealBasicVSR) produce over-smoothed results
- Diffusion models generate realistic textures and details
- But: inherent randomness causes flickering between frames

---

## Slide 3: Background — Latent Diffusion Models (LDM)

- Operate in compressed latent space (VAE encoder/decoder)
- Forward process: gradually add Gaussian noise to latent z
- Reverse process: U-Net denoiser predicts noise, iteratively denoises
- Conditioned on: LR image, text prompts, noise level

**For VSR:** Use pretrained SD x4 Upscaler (Stable Diffusion) as base model
- Already trained on image upscaling — strong prior for detail generation
- Challenge: adapt from single image to video while keeping temporal consistency

---

## Slide 4: Upscale-A-Video — Architecture Overview

**Local-Global temporal strategy within LDM framework:**

| Component | Scope | Purpose |
|-----------|-------|---------|
| Temporal U-Net | Local (8 frames) | Intra-segment consistency |
| VAE-Decoder with 3D convolutions | Local | Reduce texture flickering |
| Recurrent Latent Propagation | Global (full video) | Cross-segment consistency |
| Text prompts + noise level | Control | Quality-fidelity trade-off |

---

## Slide 5: Upscale-A-Video — Local Consistency (Temporal U-Net)

**Problem:** Pretrained SD x4 Upscaler processes images independently

**Solution:** Inflate 2D to 3D by adding temporal layers to U-Net
- **Temporal attention layers** with Rotary Position Embedding (RoPE) for time
- **3D residual blocks** (ResBlock3D) with convolutions along temporal dimension
- Only temporal layers are trained; pretrained spatial layers are frozen

**Training:**
- Freeze spatial layers, train only temporal layers
- L2 loss + LPIPS perceptual loss + temporal PatchGAN discriminator
- Training data: WebVid10M + YouHQ (37K HD clips from YouTube)

---

## Slide 6: Upscale-A-Video — Local Consistency (VAE-Decoder)

**Problem:** Even with temporal U-Net, the VAE decoder (trained on images) introduces low-level flickering when decoding latent sequences

**Solution:** Finetune VAE-Decoder with temporal modules
- Insert 3D residual blocks into decoder
- Add Spatial Feature Transform (SFT) layer conditioned on input video
  - Provides low-frequency information (color, structure) from LR input
  - Strengthens color fidelity of output
- Train with: L1 + LPIPS + GAN loss + frame difference loss

---

## Slide 7: Upscale-A-Video — Global Consistency (Recurrent Latent Propagation)

**Problem:** Temporal layers in U-Net only see local segments (8 frames). No consistency across segments.

**Solution:** Training-free flow-guided recurrent latent propagation

**How it works:**
1. Compute optical flow between frames using RAFT
2. Check forward-backward consistency to create occlusion mask M
3. At selected diffusion steps T*, propagate latents bidirectionally:
   - Warp previous frame's latent to current frame using flow
   - Fuse: blend warped latent with current latent (beta=0.5)
   - Only propagate in non-occluded regions
4. Applied at user-specified diffusion steps (e.g., steps 24, 26, 28)

**Key advantage:** No training required, works at any video length

---

## Slide 8: Upscale-A-Video — Additional Controls

**Text prompts:**
- Optional text descriptions guide texture generation
- Uses LLaVA (vision-language model) to auto-caption first frame
- Classifier-Free Guidance (CFG) enhances prompt effect
- Example: "A koala on the tree" generates more detailed fur texture

**Noise level:**
- Controls balance between restoration and generation
- Low noise: preserves input content (higher fidelity)
- High noise: generates more details (higher quality, less faithful)
- Adjustable at inference time

---

## Slide 9: MGLD-VSR — Architecture Overview

**Two key innovations:**

1. **Motion-Guided Diffusion Sampling** — Use optical flow to guide the latent denoising process for temporal consistency
2. **Temporal-Aware Sequence Decoder** — Replace standard VAE decoder with a video-aware decoder trained with sequence-oriented losses

---

## Slide 10: MGLD-VSR — Motion-Guided Diffusion Sampling

**Problem:** Standard diffusion sampling produces independent noise per frame, causing temporal flickering in latent space

**Solution:** Guide the sampling process using motion information from LR video

**How it works:**
1. Compute optical flow between LR frames
2. At each diffusion timestep, compute **warping error** of latents:
   - Warp each latent frame to its neighbors using optical flow
   - Measure L1 distance between warped and actual neighbor latents
3. Estimate **occlusion mask** M to ignore occluded regions
4. Update latent with gradient of motion-guided loss:
   - z_hat = DDPM(z, t) - eta * sigma^2 * gradient(M * E(z))

**Key insight:** Gradients of the warping error push neighboring latents to be consistent when warped by optical flow. eta=10 controls guidance strength.

---

## Slide 11: MGLD-VSR — Temporal-Aware Sequence Decoder

**Problem:** Even with motion-guided sampling, VAE decoder (8x spatial compression) introduces discontinuities in pixel space

**Solution:** Build a temporal-aware decoder on top of the pretrained VAE

**Architecture:**
- Pretrained VAE decoder (frozen spatial blocks)
- Added: 1D temporal convolutions along time dimension
- Controllable Feature Warping (CFW) module using VAE encoder features
- Interleaved spatial-temporal processing for continuity

**Training losses:**
- L_recon: L1 + LPIPS reconstruction loss
- L_diff: Frame difference loss (temporal smoothness)
- L_swc: Structure-weighted consistency loss (flow warping on GT, weighted by edge map)
- L_GAN: Adversarial loss with occlusion masking
- Total: L_video = L_recon + 0.5 * L_diff + 0.5 * L_swc + 0.025 * L_GAN

---

## Slide 12: MGLD-VSR — Training Strategy

**Two-stage training:**

**Stage 1: Denoising U-Net fine-tuning**
- Initialize from Stable Diffusion V2.1
- Insert 1D temporal convolutions for temporal modeling
- Freeze SD spatial weights, train only conditioning + temporal modules
- Batch size 24, sequence length 6, latent 64x64

**Stage 2: Temporal-aware sequence decoder**
- Generate clean latent sequences with motion-guided sampling
- Fine-tune decoder with LR sequence + generated latent + HR sequence
- Fix VAE decoder, only train temporal convolutions + CFW modules
- Batch size 4, sequence length 5, image size 512x512

---

## Slide 13: Comparison of Approaches

| Aspect | Upscale-A-Video | MGLD-VSR |
|--------|----------------|----------|
| **Base model** | SD x4 Upscaler | Stable Diffusion V2.1 |
| **Local temporal** | Temporal attention + ResBlock3D in U-Net | 1D temporal conv in U-Net |
| **Global temporal** | Recurrent latent propagation (training-free) | Motion-guided loss (gradient-based) |
| **Decoder** | Finetuned VAE-Dec with 3D conv + SFT | Temporal decoder with CFW + sequence losses |
| **Temporal guidance** | Optical flow for latent propagation | Optical flow for sampling guidance |
| **Text control** | Yes (LLaVA auto-caption + CFG) | No |
| **Noise control** | Yes (adjustable noise level) | No |
| **Training data** | WebVid10M + YouHQ (37K) | REDS (merged train+val) |
| **Venue** | CVPR 2024 | ECCV 2024 |

---

## Slide 14: Key Novelty — Upscale-A-Video

1. **Local-global temporal strategy in LDM:** First work to systematically address temporal consistency at both U-Net level (local) and across segments (global) within latent diffusion for VSR

2. **Training-free recurrent latent propagation:** Bidirectional flow-guided propagation that works at any video length without additional training — practical for long videos

3. **Text-guided VSR with controllable generation:** Text prompts and noise levels allow users to control the quality-fidelity trade-off at inference time

---

## Slide 15: Key Novelty — MGLD-VSR

1. **Motion-guided diffusion sampling:** Instead of post-hoc temporal filtering, directly guides the diffusion denoising process using optical flow gradients — temporal consistency is built into the generation process itself

2. **Sequence-oriented decoder training:** Novel losses (L_diff, L_swc) specifically designed for video sequences, with structure-weighted consistency that focuses on preserving edges and patterns

3. **Occlusion-aware guidance:** Both the sampling guidance and decoder losses use occlusion masks to avoid enforcing consistency in regions where flow estimation is unreliable

---

## Slide 16: Limitations and Connection to Our Research

**Shared limitations of both methods:**
- Process video in **short segments** (5-8 frames) — limited long-range consistency
- Memory-intensive — cannot process long videos (>1 min) in one pass
- Temporal modules scale linearly with segment length
- No explicit long-range memory or state across distant frames

**Connection to our thesis:**
- Both methods struggle with long videos due to segment-based processing
- State-space models (SSMs) offer linear-complexity long-range dependencies
- Research direction: Combine SSM temporal modeling with diffusion-based SR
- Goal: Maintain quality of diffusion models while scaling to >1 minute videos

---

## Slide 17: Experimental Setup (Our Reproduction)

**Degradation pipeline:** RealBasicVSR (blur + noise + JPEG + video compression + downscale)
- Standard for both papers
- Applied to GT to generate LQ test data

**Evaluation metrics:**
- Full-reference (synthetic): PSNR(Y), SSIM(Y), LPIPS, DISTS, VMAF
- No-reference (real-world): NIQE, BRISQUE, MUSIQ, DOVER
- All computed via pyiqa library

**Datasets:**
- Synthetic: YouHQ40, UDM10, SPMCS, REDS4/REDS30
- Real-world: VideoLQ (50 clips, no GT)

---

## Slide 18: Summary

| | Upscale-A-Video | MGLD-VSR |
|---|---|---|
| **Approach** | Local-global temporal strategy | Motion-guided diffusion + temporal decoder |
| **Temporal consistency** | Latent propagation (global) + temporal layers (local) | Gradient guidance (latent) + sequence losses (decoder) |
| **Strength** | Text/noise control, long video support via propagation | Strong perceptual quality, principled motion guidance |
| **Weakness** | No explicit motion modeling in denoising | No user controls, complex two-stage training |
| **Best for** | Controllable, practical VSR | High perceptual quality on real-world video |
