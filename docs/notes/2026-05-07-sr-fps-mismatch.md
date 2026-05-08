# SR-output FPS mismatch with LQ source

**Date:** 2026-05-07
**Affects:** all 5 synthetic test videos.

## Finding

Both SR pipelines produced exactly the same number of frames as the LQ source for every video — frame counts are identical end to end (5000, 5000, 5000, 2412, 5000). Resolution is also clean (uniform 4× upscale, 320×180 → 1280×720). But the fps tag in the output mp4 metadata is **hard-coded by each method** rather than read from the LQ source:

| video | LQ fps | MGLD fps | UAV fps | LQ source path on server |
|-------|--------|----------|---------|--------------------------|
| 7WHI2L_FDNg | 29.97 | 30.00 | 24.00 | `synthetic_data/synthetic/7WHI2L_FDNg.mkv` |
| BrRLKMbBTYQ | **24.00** | 30.00 | 24.00 | `synthetic_data/synthetic/BrRLKMbBTYQ.mkv` |
| KZ8p6b1zJ9U | 29.97 | 30.00 | 24.00 | `synthetic_data/synthetic/KZ8p6b1zJ9U.mkv` |
| hhszUXL1Cu8 | 29.97 | 30.00 | 24.00 | `synthetic_data/synthetic/hhszUXL1Cu8.mkv` |
| mJog8DlRk_4 | **23.98** | 30.00 | 24.00 | `synthetic_data/synthetic/mJog8DlRk_4.mkv` |

So MGLD always tags 30.00 fps; UAV always tags 24.00 fps. Each is wrong for some of the videos.

## Why this matters for evaluation

Frame-count identical → per-frame metrics (Human_Anatomy raw scores) are unaffected. Each frame is processed once regardless of fps tag.

Time-based metrics are affected:

- **Slow-fast Human_Identity adapter** splits at "2 seconds at native fps." On `KZ8p6b1zJ9U` MGLD's 30.00-tagged file produces 60-frame clips and 83 total clips; UAV's 24.00-tagged file produces 48-frame clips and 104 total clips. The fast branch then sees a different number of cross-clip transitions per method. The within-clip slow scores are unchanged in shape (each clip is still a chunk of the same source), but the cross-clip fast score is computed on different-sized synthetic videos.
- **tOF / tLP** and any temporal-window metrics inherit the same issue — different "windows" per method.

## Per-video score vs mismatch pattern

User observation: UAV underperforms on the videos where its fps (24) doesn't match the LQ source, and MGLD's mismatched videos (30 vs LQ 24/23.98) also have problems. The full pattern, with mismatch (✗) vs match (✓) and which method wins each metric:

| video | LQ fps | MGLD match? | UAV match? | Identity winner | Anatomy winner |
|-------|--------|-------------|------------|-----------------|----------------|
| 7WHI2L_FDNg | 29.97 | ✓ (30) | ✗ (24) | MGLD | MGLD |
| BrRLKMbBTYQ | 24.00 | ✗ (30) | ✓ (24) | MGLD | MGLD |
| KZ8p6b1zJ9U | 29.97 | ✓ (30) | ✗ (24) | **UAV** | **UAV** |
| hhszUXL1Cu8 | 29.97 | ✓ (30) | ✗ (24) | MGLD | MGLD |
| mJog8DlRk_4 | 23.98 | ✗ (30) | ✓ (24) | MGLD | MGLD |

The mismatch alone does not predict the winner — MGLD wins on `BrRLKMbBTYQ` and `mJog8DlRk_4` despite its mismatch, and UAV's mismatched videos are split (UAV wins KZ but loses the other two). So fps mismatch is **not the full story**, but it is a confound: every mismatched video has the slow-fast adapter producing wrong clip boundaries (clip duration in real-video time gets stretched or compressed). For an apples-to-apples comparison the mismatch must be removed before reading anything else into the score patterns.

## Right thing to do

Re-mux each SR output to match the LQ source's fps. Frame counts are already identical (no resampling needed) — this is a metadata change that the slow-fast adapter will read via `cv2.VideoCapture.get(CAP_PROP_FPS)` to set its clip boundaries.

`ffmpeg` isn't on `$PATH` on the lab server, but `imageio-ffmpeg` ships a bundled binary at:

```
/data/disk2/timur/miniconda3/envs/vbench/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
```

Lossless re-mux (no frame data touched, just container fps tag):

```bash
ffmpeg -r 29.97 -i mgld_synthetic_mp4/KZ8p6b1zJ9U.mp4 -c copy mgld_synthetic_mp4_fps_fixed/KZ8p6b1zJ9U.mp4
```

The `-r` before `-i` reinterprets the frame rate; `-c copy` preserves the encoded video bitstream. (cv2-based re-encoding via `cv2.VideoWriter(*'mp4v')` would be lossy — `ffmpeg -c copy` is correct.)

## Status

- 2026-05-07: documented.
- 2026-05-07 evening: tried `ffmpeg -r N -i src -c copy dst` re-mux. **It silently failed** — `-r` before `-i` only sets input rate; with `-c copy` the output mp4 inherits the original fps tag from the bitstream's timing. Verified by re-probing the "fps_fixed" videos: MGLD still 30, UAV still 24. The identity slow-fast scores on the "fps_fixed" videos were unchanged (deltas ≈ 0).
- 2026-05-08: switched to a code-level fix — added `--fps_overrides JSON` flag to `human_identity_long.py` so the splitter uses an overridden fps instead of `cv2.CAP_PROP_FPS`. Re-ran on the original videos with the LQ-derived fps per video.

## Result of the fps-corrected re-eval

**The KZ8p6b1zJ9U identity score-flip was an fps-mismatch artefact.** Old per-video pattern showed MGLD winning 4/5 and UAV winning only KZ. After fps fix:

| Video | MGLD fused (old → new) | UAV fused (old → new) | Winner (old → new) |
|-------|------------------------|------------------------|--------------------|
| 7WHI2L_FDNg | 0.366 → 0.366 | 0.337 → 0.341 | MGLD → MGLD |
| BrRLKMbBTYQ | 0.756 → 0.760 | 0.481 → 0.481 | MGLD → MGLD |
| **KZ8p6b1zJ9U** | 0.657 → 0.657 | **0.751 → 0.629** | **UAV → MGLD** |
| hhszUXL1Cu8 | 0.655 → 0.655 | 0.460 → 0.561 | MGLD → MGLD |
| mJog8DlRk_4 | 0.341 → 0.346 | 0.285 → 0.285 | MGLD → MGLD |
| Mean | 0.555 → 0.557 | 0.463 → 0.459 | MGLD +0.092 → MGLD +0.097 |

What actually changed:
- KZ UAV fast-branch score dropped from 0.778 → 0.579 because both methods now produce 83 clips (instead of MGLD 83 vs UAV 104), so UAV's fast branch sees fewer cross-clip transitions and ArcFace embedding drift is not artificially smoothed.
- KZ MGLD essentially unchanged (its 30-fps tag was already close to LQ 29.97).
- Other videos: small changes in the right direction (UAV gains slightly where its 24 was wrong, MGLD gains slightly where its 30 was wrong).

**With fps-corrected splitting, MGLD wins Human_Identity on all 5/5 videos.**

**On Anatomy and fps**: the upstream `Human_Anatomy` *as we ran it* (whole-video `custom_input` mode) is fps-invariant — it iterates frames independently, computes per-frame abnormality, and aggregates as `1 - sum_frames(abnormal) / sum_frames(people)`. No clip windowing, no fps used. So the existing whole-video Anatomy numbers (MGLD 0.600 vs UAV 0.605 mean, KZ MGLD 0.144 vs UAV 0.435) are unchanged by the fps fix.

But this is asymmetric with how we evaluate Identity. Identity uses a slow-fast wrapper that *is* fps-dependent. The right symmetric setup is a **slow-fast Anatomy** wrapper too — split per-frame trace into 2-sec clips at LQ-fps, average per-clip abnormal-rate (slow), pool first-frames-of-clips and score them as one synthetic video (fast), fuse 50/50. That wrapper would be fps-dependent in the same way Identity is, and would be the apples-to-apples Anatomy comparison for long videos.

We have per-frame Anatomy traces cached for KZ8p6b1zJ9U and hhszUXL1Cu8 already, so the slow-fast Anatomy aggregation on those two is a post-hoc Python compute (no GPU). For the other 3 videos we need to run `diagnose_anatomy_per_frame.py` first to get the per-frame trace.

The KZ anatomy outlier (UAV 0.435 vs MGLD 0.144 in whole-video mode) still stands as a genuine metric-failure case at the **per-frame level** — the regime-shift finding in `docs/plans/2026-05-07-metric-failure-diagnostic.md` is independent of any windowing scheme. Whether slow-fast Anatomy on KZ also flips to UAV-wins (as identity did before fps fix) is what the new aggregator will tell us.
