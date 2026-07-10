# Design — RoPE Position-Embedding Extrapolation in Long-Video VSR

**Date:** 2026-07-02
**Author:** Timur Iakshibaev (with Claude)
**Status:** Design — awaiting review
**Relation to thesis:** Post-July-15 research probe. Maps to **Direction 4** of
`docs/plans/2026-06-15-long-term-plan.md` (long-context / positional modelling
for long-video SR). NOT part of the July 15 blind-review thesis (which is frozen
July 1 and centred on LR-VCC). Intended to seed the paper co-write / next arc.

---

## 1. Motivation and hypothesis

Diffusion-transformer VSR models encode frame position with **rotary position
embeddings (RoPE)**. RoPE is *relative* by design, but in practice absolute
position magnitude and boundary effects leak into attention, and RoPE is known
(from the LLM long-context literature) to **degrade when it must extrapolate
beyond its trained position range**. Long videos (>1 minute) are exactly the
regime that forces such extrapolation.

**Central hypothesis (H):** For a RoPE-based VSR model, quality degradation on
long videos is caused in part by attention/RoPE **failing to extrapolate** to
temporal positions beyond its training range — not only by content difficulty.

**Sub-claims:**
- **H0 (control):** If attention were perfectly relative, shifting all temporal
  position indices by a constant `k` (content fixed) would not change the output.
  We expect small-but-nonzero drift → absolute-position leakage exists.
- **H1 (phenomenon):** Forcing temporal positions beyond the trained range
  (content fixed) degrades output quality monotonically with position magnitude
  / sequence length.

This pairs naturally with the thesis's existing asset: **LR-VCC** already
*measures* long-range consistency degradation; this study probes a candidate
*cause*.

## 2. Scope

**In scope (this spec):**
- **One model: FlashVSR** (OpenImagingLab). Diffusion transformer on the
  **Wan2.1** backbone (ships `Wan2.1_VAE`); Wan2.1 DiT uses **3D RoPE**
  (temporal + H + W). Explicitly a streaming, long-video model (`tiny_long_video`
  inference path), ~17 FPS @ 768×1408 on one A100 → fits the 40 GB cards.
- **Two perturbation levers:** temporal index **shift** (control) and temporal
  **length/position extrapolation** (phenomenon). Content held fixed in both.
- **Three references:** self-consistency (perturbed vs unperturbed output),
  absolute quality vs HR GT, and **LR-VCC** (long-range consistency).
- **A thin mechanism diagnostic:** per-layer attention/activation drift logged
  from the same runs (cheap "why", not a separate study).

**Deferred (explicitly out, revisit only if H1 confirmed):**
- **SeedVR2** (ByteDance-Seed/SeedVR, Apache 2.0; 3B and 7B one-step DiT).
  Primary round-2 contrast (decided 2026-07-02): uses **adaptive window
  attention** (window sized to output resolution) — a window-local positional
  regime, vs FlashVSR's absolute ever-growing streaming positions. Question:
  does window-local position handling avoid long-video positional drift?
  Caveats: official inference targets H100-80G (use the 3B and/or the
  community ComfyUI port's FP8/BlockSwap optimizations on our 40 GB A100s);
  RoPE-within-windows to confirm from code.
- **SparkVSR** (CogVideoX1.5-5B-I2V, also 3D RoPE, keyframe-conditioned).
  Secondary contrast: does keyframe re-anchoring mask extrapolation drift?
  Heavier (5B) and a positional confound — lowest priority.
- **Mitigation levers:** RoPE θ-rescale / NTK-aware scaling / position
  interpolation. These are the "can we fix it" follow-up, not part of the probe.
- **Spatial RoPE perturbation** (crop/resolution/tiling seams). Different question.

## 3. Approach

**Approach A — single-model deep probe (FlashVSR), with a thin slice of
mechanism logging (C).** Confirm and instrument FlashVSR's RoPE, run shift +
extrapolation with all three references, and log a cheap attention/activation
drift diagnostic from the same forward passes. Chosen over a two-model
comparison (2× infra, 5B awkward on 40 GB, spreads thin before the effect is
known real) and over a pure mechanism study (further from the "standard metrics"
goal). SparkVSR is scheduled as round 2, gated on H1.

## 4. Architecture of the experiment

Five phases. Phase 0 is the critical path — every downstream measurement needs
the position-override handle it builds.

### Phase 0 — Confirm & instrument (FlashVSR)

**Goal:** a verified, minimally-invasive handle to (a) read and (b) override the
temporal position indices and the extrapolation length in FlashVSR's DiT, plus
a no-op check that proves the handle is faithful.

- Stand up FlashVSR inference on the SmartML server (own conda env; weights via
  `hf-mirror.com`). Reproduce a baseline SR output on a short clip.
- Locate the RoPE call site in the Wan2.1 DiT (temporal freq computation /
  `rope`/`rotary`/`freqs` construction; the temporal axis of the 3D RoPE).
- Add a hook / small patch exposing: the temporal position index tensor, a
  constant **shift** `k`, and a **position-span** override (stretch factor `s`
  and/or explicit index list) — content path untouched.
- **Faithfulness gate:** with shift `k=0` and no stretch, output must match the
  unpatched baseline **bit-for-bit** (or within fp nondeterminism floor). If it
  doesn't, the handle is wrong and Phases 1–2 are meaningless.

**Interface (unit boundary):** a single function that, given input frames and a
position-override spec `{shift:int, stretch:float|None, indices:list|None,
length:int}`, returns the SR output. All experiments call only this.

**Depends on:** FlashVSR repo + weights, server GPU. **Risk:** RoPE site harder
to isolate than expected (sparse-attention custom kernels) → see §7.

### Phase 1 — Shift control (self-consistency)

- Same `T` frames; temporal indices `[0..T-1]` (baseline) vs `[k..k+T-1]` for a
  sweep of `k` (e.g. within-range and beyond-range values).
- **Measure:** PSNR / SSIM / LPIPS of **perturbed-vs-unperturbed output**, plus
  **LR-VCC**. No GT required → runs on any video, including the 5 long synthetic
  clips already on the server.
- **Read:** near-zero drift for in-range `k` = good relative behaviour; rising
  drift as `k` pushes beyond range = absolute-position leakage / boundary effect.

### Phase 2 — Length extrapolation (the phenomenon)

Two complementary manipulations, content fixed:

- **(2a) Position stretch (cleanest isolation):** same `T` frames, temporal
  indices `[0, s, 2s, …]` for increasing `s`, forcing large positions without
  changing pixels. Sweep `s`.
- **(2b) Long single-pass vs in-range chunked:** process a long video as one
  extrapolating sequence vs matched **in-range chunks** (positions always
  in-trained-range). Compare per-frame within matched windows (overlap chunks to
  control receptive-field confound).

- **Measure:** self-consistency + **vs-GT** (where GT exists, see Phase 3) +
  **LR-VCC**, all as a function of position magnitude / length.
- **Read (H1):** quality falls monotonically once positions exceed the trained
  range; LR-VCC should register long-range consistency loss that per-frame PSNR
  may partially miss.

### Phase 3 — Data for the GT claim

- Self-consistency (Phases 1, 2a) and LR-VCC need **no GT** → use the 5 long
  synthetic videos already on the server (80–208 s) for the primary
  extrapolation-vs-length curves.
- For **quality-vs-GT** on long content (existing GT sets UDM10/SPMCS/REDS are
  short), **build a small long-HR set:** curate a handful of ~1-min 720p HR
  clips, bicubic-↓×4 to LR on the Mac, bridge LR+GT to the server via the GitHub
  branch method. Explicitly a small data task; kept minimal (3–5 clips) since it
  supports a *secondary* absolute-quality claim, not the primary curve.

### Optional diagnostic — mechanism slice (from C)

- From the same Phase 1–2 forward passes, log a cheap per-layer signal
  (attention-entropy and/or activation L2 drift vs the baseline run) as a
  function of position magnitude. Gives the mechanistic "why" (where in the stack
  extrapolation breaks) at near-zero extra compute. Purely additive; drop if it
  complicates Phase 0.

## 5. Metrics & datasets summary

| Phase | Manipulation | Reference(s) | Data | GPU |
|---|---|---|---|---|
| 0 | none (no-op) | bit-exact vs baseline | 1 short clip | yes |
| 1 | index shift `k` | self-consistency, LR-VCC | long synthetic + short | yes |
| 2a | position stretch `s` | self-consistency, vs-GT, LR-VCC | long synthetic + long-GT | yes |
| 2b | single-pass vs chunked | self-consistency, vs-GT, LR-VCC | long synthetic + long-GT | yes |
| 3 | (data build) | — | curate long-HR → LR | Mac + bridge |

Metrics: **PSNR, SSIM, LPIPS** (per-frame), **LR-VCC** (long-range consistency,
reuse `scripts/lr_vcc/`). Mechanism diagnostic: per-layer attention-entropy /
activation drift.

## 6. Deliverables

- `scripts/rope_probe/` — inference wrapper with the position-override interface
  (Phase 0), plus perturbation drivers for Phases 1–2 and an analysis/plotting
  script. Parameterised and reproducible (per the repo's ablation-script lesson).
- Result JSONs under `results/rope_probe/<phase>/<video>/…` (gitignored;
  figures git-tracked, per existing convention).
- Curves: metric vs shift `k`, metric vs stretch `s`, metric vs length; the
  chunked-vs-single-pass table; the optional per-layer drift plot.
- A short findings note in `docs/notes/` summarising whether H0/H1 hold, feeding
  the paper / Direction-4 arc.

## 7. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| RoPE site buried in custom sparse-attention kernels (LCSA) | Phase 0 blocked | Start from the standard Wan2.1 DiT RoPE (LCSA changes the *mask*, not the position construction); if the streaming path obscures it, instrument the non-streaming `tiny` path first |
| FlashVSR won't fit / run on shared 40 GB A100s | no experiments | tile/`tiny_long_video` path is built for memory; check `nvidia-smi` free mem first; cap footprint per GPU-etiquette doc |
| No-op override not bit-exact (nondeterminism) | can't trust drift signal | fix seeds, disable nondeterministic kernels, establish an fp noise floor and report drift relative to it |
| Long-HR GT curation slow / blocked (YouTube blocked on server) | weaker vs-GT claim | curate on the Mac, bridge; keep to 3–5 clips; primary evidence is self-consistency + LR-VCC which need no GT |
| Mac↔server transfer unusable (~22 KB/s) | can't move data | GitHub-branch bridge (documented in `docs/2026-07-01-new-server-and-gotchas.md`) |
| Effect is null (H1 false) | probe "fails" | still a publishable negative + the shift-control result; pivot to spatial RoPE or SparkVSR keyframe contrast |

## 8. Success criteria

- **Phase 0 passes** (faithful, bit-exact no-op override) — non-negotiable gate.
- Phases 1–2 produce **clean metric-vs-position curves** with all three
  references on ≥ the 5 long synthetic videos + a small long-GT set.
- A clear **verdict on H0 and H1**, with LR-VCC vs PSNR/SSIM contrast quantified.
- Enough signal to decide the round-2 fork: **SparkVSR contrast** and/or the
  **θ-rescale mitigation** lever.

## 9. Open decisions (for the plan step)

1. FlashVSR env: fresh conda env vs reuse `vbench` (torch 2.5.1). Likely fresh,
   pinned to FlashVSR's requirements.
2. Exact `k` and `s` sweep grids and the model's trained temporal range (read
   from config once inference is up).
3. Which/how many long-HR clips to curate for the GT set (defer until Phase 2a
   self-consistency confirms the effect is worth the GT effort).
