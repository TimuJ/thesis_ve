# Progress Report — June 29 – July 12, 2026

**Topic:** Video Super-Resolution for Long Videos — LR-VCC benchmark + RoPE extrapolation probe

## Headline

**FlashVSR tops the 3-method LR-VCC table (0.610 > MGLD 0.589 > UAV 0.552) while
showing the worst long-range drift cell (D″) — and the RoPE probe delivered its causal
verdict: that drift is NOT positional.** Absolute RoPE offsets are quality-free even
50× beyond the trained window on every axis; what damages quality is relative-distance
geometry distortion (dilation), while mild PI-compression is free. Position encoding is
exonerated — long-video improvement effort belongs to the streaming cache/generation
mechanism and resolution-scaling mechanics. The verdict matrix also reached **12/12**
(flip_invert control completed: PASS on all 5 bases), and thesis writing is now the
foreground: a full 7-chapter skeleton builds clean on `main`.

## Key Results

### 1. Real-model LR-VCC v5 — the 3-method table

| video | MGLD | UAV | FlashVSR | winner |
|---|---:|---:|---:|:--:|
| 7WHI | 0.652 | 0.609 | **0.737** | FlashVSR |
| BrRLK | **0.407** | 0.380 | 0.393 | MGLD |
| KZ | **0.732** | 0.712 | 0.722 | MGLD |
| hhsz | **0.579** | 0.528 | 0.550 | MGLD |
| mJog | 0.575 | 0.533 | **0.649** | FlashVSR |
| **mean** | 0.589 | 0.552 | **0.610** | **FlashVSR** |

FlashVSR's signature: **best Identity** (0.598 vs MGLD 0.555 vs UAV 0.463) but **worst
D″ CLIP-trajectory** (0.862 vs MGLD 0.893) — exactly the long-range-drift cell the new
research arc investigates. FlashVSR was stood up from zero on the new server
(pristine-tagged repo, staged setup scripts).

### 2. RoPE extrapolation probe — from idea to causal verdicts in 10 days

Bit-exact position-injection hook (no-op drift 0.0 on a measured 0.0 floor, 48 unit
tests, zero modification of the FlashVSR repo) → shift/stretch/continuous-PI sweeps →
vs-GT verdicts on DOVE-UDM10 (10 clips × 22 conditions; baseline 24.02 dB vs MGLD's
verified 24.23):

| condition | GT cost |
|---|---:|
| shift +996 (50× out-of-window) | **+0.001 dB — free** |
| PI-compression s=0.75 | **+0.01 dB — free** |
| dilation s=1.25 / 1.5 / 2.0 / 3.0 | −0.12 / −0.22 / −0.25 / −0.95 dB |
| compression s=0.5 | −0.66 dB |

Extras: the stock 4089-frame single-pass ceiling is an implementation artefact of the
fixed RoPE table — the extended table sustained a **5009-frame single pass** (11.6 GiB
VRAM); and **self-consistency provably fails to predict quality** (s=0.75 vs s=1.5:
identical self-PSNR, opposite GT verdicts) — position-perturbation studies need real
references.

### 3. D″ causal check — FlashVSR's long-video drift is not positional

Three arms on the 2412–5000-frame videos — segmented / single-pass / magnitude-bounded
(mod 336) — are **indistinguishable (<1 % relative spread)**. The drift belongs to the
streaming cache/generation mechanism, not position encoding; it also certifies the §1
benchmark row was fair to FlashVSR.

### 4. Comprehensive time×space sensitivity matrix (group ask, July 10) — [standalone report](rope_timespace_sensitivity_matrix.md)

Translation invariance holds on **all three RoPE axes**; a PI-free zone (mild
compression) exists on every axis; space is ~2.5× more sensitive than time at extreme
geometry distortion. Real resolution extrapolation on YouHQ40: 1.5× grid growth costs
only **−0.15 dB stock**, while spatial-PI there *hurts* (−0.56) — do not apply PI at
≤1.5× spatial extension. The 1440² rung collapses (−11 dB) for **non-positional**
reasons (adaptive topk sparsity is the prime suspect; follow-up defined).

### 5. Verdict matrix completed — 12/12

The flip_invert identity stage (killed by the June 15 server decommission) was rebuilt
on the new server and completed July 2: **PASS on all 5 bases**, exactly as the
positive-control prediction said. Full v5 matrix now 12 artefacts × 5 bases, clean
(PASS+WEAK) **28/60**.

### 6. Thesis track — writing is now the foreground

Blind-review deadline corrected to **July 25 (hard)**. The `rope-probe` branch was
merged to `main` (single-branch workflow) and a full 7-chapter thesis skeleton is in
place and building clean (66 pp): Ch1 Introduction+Related Work, Ch3 failure modes, and
Ch4 LR-VCC carry near-final prose (proposal ports upgraded to v5); Ch2/5/6/7 are
structured skeletons with every number traced to its canonical source file.

### 7. Infrastructure note (one line)

Two disk-at-100% crises weathered with zero data loss (GitHub-bridge pattern, scoped
7-day PAT — lapsed on schedule July 9); YouTube long-GT re-acquisition is **closed**
(platform enforcement, all clients/networks) — the long-GT path forward is lab-provided
HR footage via `make_long_gt.py` (group ask pending).

## Code delivered this period

| File | Purpose |
|---|---|
| `scripts/rope_probe/` (hook, sweep drivers, causal arms, ladder scorer) | position-injection instrument, 48 unit tests |
| `results/rope_probe/` (shift, stretch, PI, udm10_gt, dpp_causal, spatial h/w, res_ladder) | all probe result JSONs |
| `docs/notes/2026-07-02-flashvsr-rope-site.md` | RoPE site analysis + faithfulness gates |
| `docs/notes/2026-07-11-rope-extrapolation-findings.md` | probe capstone — 5 verdicts, handoff, limitations |
| `reports/figures/` (rope_sensitivity_matrix, rope_probe_udm10_gt, dpp_causal_verdict, realmodel_v5_3method) | verdict tables |
| `scripts/finalize_flip_invert.sh` | one-command matrix completion (merge → compose → rebuild) |
| `docs/2026-07-01-new-server-and-gotchas.md` | new-server restore guide |
| `zjuthesis/body/graduate-eng/` (7 chapters) | thesis skeleton, builds clean |

Test suite **174 passing** (126 previous + 48 probe), 0 failing.

## Next Period (July 13 – 26)

Thesis writing is the critical path; probe work continues only as background handoff.

1. **Thesis chapters to blind review.** Methodology (Jul 13–15) → experiments (15–17)
   → RoPE study (17–19) → related work / foundations polish (19–21) → conclusion +
   abstracts EN/中文 (21–22) → consistency pass (23) → blind-review build (24).
2. **Window-extension handoff.** Predictions (temporal ~1.33× with continuous-PI
   predicted RoPE-loss-free) + validated hook are with the window-extension student;
   support their extended-window runs.
3. **Optional rigour ablations** if time allows: β/τ sensitivity sweep + leave-one-out
   sub-metric attribution — pure local recompute from cached JSONs, no GPU.
4. **SeedVR2 round-2 contrast** (window-local vs absolute streaming positions) —
   post-deadline backlog.

**Blind-review thesis submission: July 25 (hard).**
