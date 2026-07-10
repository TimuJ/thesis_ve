# Biweekly Report — Timur Iakshibaev

## Period: June 29 – July 12, 2026 (first biweekly)

> New cadence: biweekly reports aggregate the weekly ones. Sources for this
> period: `Timur_Iakshibaev_2026-06-29_to_2026-07-01.md` (extended through
> July 3 + addenda) and `Timur_Iakshibaev_2026-07-04_to_2026-07-10.md`.

## The period in three headlines

1. **LR-VCC's real-model story got its centrepiece and then its third row:**
   the v5 metric ranks MGLD > UAV unanimously (thesis headline experiment),
   and FlashVSR — stood up from zero on the new server — now **tops the
   3-method table** (0.610 vs 0.589 vs 0.552), winning on identity while
   showing the worst long-range-drift cell (D″), exactly the signature the
   new research arc investigates.
2. **The RoPE-extrapolation probe went from idea to verified instrument to
   completed causal verdicts:** spec → 10-task plan → bit-exact
   position-injection hook (no-op drift 0.0 on a 0.0 floor) → shift/stretch/
   continuous-PI sweeps → vs-GT run on DOVE-UDM10 → the D″ causal check.
   Verdicts: absolute position magnitude is quality-free even 50× beyond
   the trained window; mild PI-compression (s=0.75) is quality-free;
   dilation damages monotonically (with the important caveat that the small
   costs at s≤2 are still *inside* the trained distance range under
   streaming's ~8-latent span — the first out-of-window point, s=3, costs
   −0.95 dB); self-consistency provably fails to predict quality; and
   **FlashVSR's long-video CLIP-trajectory drift is NOT positional** (three
   arms — segmented / single-pass / magnitude-bounded — indistinguishable,
   < 1 % relative spread), exonerating RoPE and pointing long-video drift
   fixes at the streaming/caching mechanism instead.
3. **Infrastructure held despite two crises** (server disk at 100%, twice;
   Google-Drive and YouTube transfer walls) thanks to the GitHub-bridge
   pattern — now bidirectional via a scoped 7-day PAT — and everything is
   reproducible: pristine-tagged FlashVSR repo, staged setup scripts,
   regression-tested probe code (40 local tests).

## Key numbers

| result | value |
|---|---|
| LR-VCC v5 3-method mean | FlashVSR **0.610** > MGLD 0.589 > UAV 0.552 |
| FlashVSR vs DOVE-UDM10 GT (baseline) | 24.02 dB (MGLD verified: 24.23) |
| Shift 996 (50× out-of-window) GT cost | **+0.001 dB — free** |
| PI-compression s=0.75 GT cost | **+0.01 dB — free** |
| Dilation s=1.25 / 2.0 / 3.0 GT cost | −0.12 / −0.25 / −0.95 dB |
| Stock FlashVSR single-pass ceiling | 4089 frames (~2.3 min) — RoPE table limit |
| …removed by the probe's extended table | 5009-frame single pass, 11.6 GiB VRAM, works |
| D″ causal check (segmented / single / mod336) | indistinguishable, < 1 % — **drift not positional** |
| Probe faithfulness gate | bit-exact (floor 0.0, no-op 0.0) |

## Decisions and framing this period

- **Supervisor confirmation (July 3):** main goal = how RoPE extrapolates;
  group frames it via the 21-latent (~81-frame) trained window; another
  student extends the window. Our decomposition (magnitude free / distance
  bites) plus the continuous-PI tool is the direct handoff.
- **SeedVR2** replaces SparkVSR as the round-2 contrast model (window-local
  positions vs FlashVSR's absolute streaming positions).
- **Blind-review deadline corrected: July 25 (hard)** — thesis writing takes
  the foreground from ~July 8; probe continues as background GPU jobs.
- Evaluation stays on the DOVE protocol (April decision honoured for the
  probe's GT set choice: UDM10 with realistic degradation over REDS bicubic).

## Week-2 outcomes (July 7–12)

- [x] **D″ causal check — done, decisive:** drift is not positional
      (`reports/figures/dpp_causal_verdict.md`); benchmark row fair;
      stock frame ceiling proven removable in production.
- [x] vs-GT quality verdict on DOVE-UDM10 (10 clips × 22 conditions);
      continuous-PI tool built and validated.
- [x] **Comprehensive time×space sensitivity matrix — done** (group ask,
      July 10): spatial (h/w) label sweeps on UDM10 + a real
      resolution-extrapolation ladder on YouHQ40 (grids 45²→72²→90², stock
      vs spatial-PI). Headlines: translation invariance universal across all
      3 RoPE axes; the PI-free zone exists on every axis; space ~2.5× more
      sensitive than time at extreme geometry distortion; real 1.5× grid
      extension costs only −0.15 dB (RoPE exonerated again) while spatial-PI
      there *hurts* (−0.56); the 1440² rung collapses for non-positional
      reasons (adaptive sparsity / input confound — follow-up defined).
      Full matrix: `reports/figures/rope_sensitivity_matrix.md`.
- [ ] Extended-window experiment with the window-extension student —
      predictions + tool handed over; next: their runs.
- [x] Task 10 findings note — probe capstone written
      (`docs/notes/2026-07-11-rope-extrapolation-findings.md`); sensitivity
      tables serve as the Task-8 deliverable. Remaining: whole-branch review
      before merging `rope-probe`.
- [x] Long-video GT via YouTube: **exhausted and closed** — platform-side
      enforcement blocks all clients on all networks tried (campus, mobile,
      cookies, PO-token provider). Path forward: lab-provided long HR
      footage (group ask) via `make_long_gt.py` — cleaner GT anyway.
- [x] PAT lapsed ~July 9 as planned; remaining artefact traffic is
      JSON-sized (scp suffices).
- [ ] **Methodology + experiments chapter drafts** (thesis, July 25) — the
      foreground from here.

## One-line summary for the meeting

FlashVSR's RoPE extrapolates benignly in every dimension we can measure —
offsets are free on all three axes, moderate real extension (temporal
single-pass ×2.5 length, spatial ×1.5 grid) costs ≲0.15 dB, and the failure
modes that do exist (long-video drift, ≥1440² collapse) are demonstrably not
positional. Position-side fixes (PI) help only in narrow regimes and can
hurt; effort should target streaming cache/generation and
resolution-scaling mechanics instead.
