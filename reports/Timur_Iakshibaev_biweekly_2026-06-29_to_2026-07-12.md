# Biweekly Report — Timur Iakshibaev

## Period: June 29 – July 12, 2026 (first biweekly; IN PROGRESS — finalise July 12)

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
   first quality verdicts in five days:** spec → 10-task plan → bit-exact
   position-injection hook (no-op drift 0.0 on a 0.0 floor) → shift/stretch/
   continuous-PI sweeps → vs-GT run on DOVE-UDM10. Verdicts: absolute
   position magnitude is quality-free even 50× beyond the trained window;
   mild PI-compression (s=0.75) is quality-free; dilation damages
   monotonically; self-consistency provably fails to predict quality.
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

## Open items into week 2 (July 7–12)

- [ ] Extended-window experiment with the window-extension student
      (predictions + tool ready).
- [ ] D″ causal check (position-reset vs stock on a long video, LR-VCC-scored).
- [ ] Task 8 analysis curves + Task 10 findings note; whole-branch review.
- [ ] Long-video GT: YouTube re-download from an unflagged network (parked).
- [ ] PAT expiry ~July 9: finish bulk transfers or mint a successor.
- [ ] **Methodology + experiments chapter drafts** (thesis, July 25).

_To be finalised with week-2 outcomes on July 12._
