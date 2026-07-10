# Weekly Progress Report — Timur Iakshibaev

## Period: July 4 – 10, 2026

## Headline

The RoPE-extrapolation probe went from "instrument verified" to **first
quality-bearing scientific verdicts**. Three sweeps (fine/fractional stretch,
continuous-position PI, and the decisive vs-ground-truth run on DOVE-UDM10)
established: **(1)** absolute-position extrapolation is *completely free* in
GT quality — even 50× beyond the trained window (H0 settled); **(2)** mild
position *compression* (s=0.75, the Position-Interpolation regime) is
quality-free, while dilation damages monotonically; **(3)** self-consistency
provably does not predict quality — two conditions with identical
self-change showed opposite GT verdicts, retroactively justifying the whole
vs-GT effort. The probe now hands the window-extension study (the other
student's task) a concrete, quality-backed prediction with a validated tool.

---

## 1. Continuous-position PI (July 4–5)

The July-4 fine-stretch sweep exposed an integer-rounding confound (compressed
positions collided into duplicates). Built true LLM-style position
interpolation: fractional positions with RoPE rows computed on the fly by the
model's own frequency formula (`default_row_builder`), validated by a
continuous-identity condition landing exactly on the 53 dB numerics floor and
by integer-multiple conditions matching table lookups to 0.01 dB.

Result: rounding explained ~half the compression-vs-dilation asymmetry in
self-consistency; true-PI compression still *changed* output about as much as
dilation — but see §2 for why "change" turned out to be the wrong metric.

## 2. The vs-GT quality verdict on DOVE-UDM10 (July 5–6)

**Data:** the April protocol decision (align with DOVE) was honoured — DOVE's
UDM10 test set (10 clips, GT 1272×720 + realistic-degradation LQ 318×180)
was bridged to the server; the probe runner gained arbitrary-LR support and
GT-crop scoring (test-covered). 10 clips × 22 conditions (full shift×stretch
cross, continuous positions) scored with pyiqa (DOVE RGB convention) against
true GT. Full tables: `reports/figures/rope_probe_udm10_gt.md`.

**Baseline sanity:** FlashVSR 24.02 dB on UDM10 — beside MGLD's verified
24.23 on the same protocol.

**Findings:**

1. **Shift is free — H0 settled in quality terms.** Positions 996–1020
   (≫ trained 0–20): ΔPSNR +0.001, SSIM/LPIPS unchanged; holds at every
   stretch level (interactions ±0.002 dB). FlashVSR's unbounded streaming
   position growth is not, by itself, a quality problem.
2. **The PI regime is quality-free; dilation is not.** s=0.75 → **+0.01 dB**
   (SSIM up); dilation monotone: −0.12 (1.25), −0.22 (1.5), −0.25 (2.0),
   −0.95 (3.0); heavy compression −0.66 (0.5).
3. **Self-consistency ≠ quality, proven:** s=0.75 vs s=1.5 — identical
   self-PSNR (31.8), opposite GT verdicts (+0.01 vs −0.22). The earlier
   sweeps measured change; only this run measured harm.

**Handoff to the window-extension study:** extending the trained 21-latent
window to ~28 latents with PI-compressed positions is predicted loss-free
from RoPE's side; stock (dilated-geometry) extension should cost ~0.1–0.25 dB
at moderate factors. Both predictions are directly testable with our hook
(continuous PI, zero model modification, bit-exact when idle).

## 3. Supervisor-framing integration (July 3–4, recap)

The group's "21-frame window / ~80 frames" was pinned to code: 81 pixel
frames = 21 latent frames = trained RoPE range; streaming keeps relative
distances ≤ ~8 latents by construction, so only absolute magnitude grows —
and §2.1 now shows that magnitude is harmless. The window-exit risk is
entirely in relative-distance geometry, which is what window extension
changes.

## 4. Infrastructure notes

- **GT acquisition:** YouTube re-download of the 5 long sources (for true
  long-video GT) is blocked by captcha/IP-flagging — parked; retry from a
  different network. DOVE-UDM10 came via browser download + GitHub bridge
  (Google Drive throttled gdown to 51 KB/s; the browser was 100×).
- Server disk crisis (100%) resolved: pip cache purge + scratch cleanup →
  156 GB free. GPUs were unusually idle July 5 (~34 GB free each) — the
  10-clip sweep ran in ~40 min.
- One process bug caught and fixed with a regression test: a refactor of the
  runner's `prepare()` had silently not applied (text-replacement mismatch),
  crashing the first UDM10 launch; the sweep loop's log filtering masked the
  traceback. Lesson recorded: no output-filtering on long-run logs, and
  server-only code paths get local regression tests.
- PAT (server→GitHub push auth) expires **~July 9** — remaining bulk
  transfers should happen before then or a new token gets minted.

## 5. The D″ causal check (July 10, in flight)

The experiment connecting the probe to the benchmark's weakest FlashVSR cell
(D″ CLIP-trajectory drift). Design: same long videos, three arms —
**A** segmented stock (the benchmark outputs, position+cache reset at frame
2500), **B** true single-pass (positions grow to latent ~1252, served by the
extended table), **C** single-pass with positions cycled mod 336 (magnitude
bounded, no content seam). B≈A≈C → drift not positional; B worse than C →
accumulated magnitude matters; B better than A → the segmentation seam was
the cost. Every outcome is informative.

Enablers built (test-covered, 45 local tests): `modulo` position transform;
a fix for a real hook bug (slice bounds were clamped to the table length —
past-the-table chunks would have silently received empty position lists);
a chunked uint8 conversion in the long-video driver after the first attempt's
5000-frame runs were OOM-killed *post-inference* by the naive whole-video
fp32 decode (~120 GB transient).

**Already established:** (1) single-pass inference past the stock 4089-frame
ceiling **works in production** — 5009 frames, latents to 1252 on the
extended table, 11.6 GiB VRAM; the ceiling is purely a stock-code artefact,
removable via positions alone. (2) Preliminary causal data point (hhsz,
2412 frames): D″ = 0.4216 stock vs 0.4220 with cycled positions —
**position magnitude does not move drift** on this video. The four
5000-frame arms (where arm A has a real seam) are re-running with the memory
fix; final A/B/C table to follow.

## 6. Next

1. Finalise the D″ causal table (runs in flight) → findings note (Task 10)
   + Task 8 curves; then the whole-branch review.
2. Extended-window experiment coordination with the window-extension student
   (their engineering + our position tool + these predictions).
3. **Thesis writing is now the foreground** (blind review July 25, hard);
   probe work continues as background GPU jobs only.
4. Housekeeping: the server GitHub PAT has lapsed (~July 9) — mint a
   successor only if another bulk transfer is needed; small files go by scp.
