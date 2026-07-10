# D″ causal check — verdict: drift is NOT positional (2026-07-10)

Same long videos, three arms, D″ (CLIP-trajectory, `compute_clip_trajectory`
defaults, higher = less drift):

- **A — segmented stock**: the benchmark outputs (position + cache reset at
  frame 2500, forced by the stock 1024-row RoPE table).
- **B — true single-pass**: positions grow to latent ~1252, served by the
  probe's extended table. No seam.
- **C — single-pass, positions cycled mod 336**: magnitude bounded, no
  content seam, rare position wraps.

| video | A segmented | B single-pass | C mod336 | max spread |
|---|---:|---:|---:|---:|
| hhszUXL1Cu8 (2412f) | 0.4216 | (≡A, single-seg) | 0.4220 | 0.0004 |
| 7WHI2L_FDNg (5000f) | 0.1680 | 0.1671 | 0.1674 | 0.0009 |
| mJog8DlRk_4 (5000f) | 0.2177 | 0.2161 | 0.2165 | 0.0016 |

## Verdict

All three arms are indistinguishable (spread ≤ 0.0016 on scores spanning
0.17–0.42 across videos — i.e. < 1 % relative). Conclusions:

1. **FlashVSR's CLIP-trajectory drift is not caused by RoPE positions** —
   not by absolute magnitude (B ≈ C), not by growth past the table (B ran to
   latent 1252), and not by the segmentation seam (B ≈ A). The drift is a
   property of the content and/or the streaming generation itself (KV-cache
   error accumulation, decoder), not of position encoding. Consistent with
   the shift-freeness results at every scale tested.
2. **The benchmark's FlashVSR row is fair:** forced segmentation did not
   penalise its D″ cell; single-pass would have scored the same.
3. For the window-extension arc this closes the "maybe long-video drift is
   positional" branch: fixes for long-video drift must target the
   generation/caching mechanism, not position encoding; conversely, position
   handling is exonerated as the streaming-length bottleneck (the table
   ceiling is an implementation artefact, removable — proven in production
   here at 5009 frames / 11.6 GiB VRAM).

Artefacts: `results/rope_probe/dpp_causal/` (server: per-arm mp4+PNGs;
local: D″ JSONs + run stats).
