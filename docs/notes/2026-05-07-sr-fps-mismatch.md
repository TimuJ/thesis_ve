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

- 2026-05-07: documented; re-muxing `KZ8p6b1zJ9U` (and the other 4) for both methods next, then re-running per-clip identity + Step 1.5 anatomy on the re-muxed files to determine whether the KZ score-flip survives the fps fix.
- TBD: re-run the full whole-video Anatomy battery on re-muxed videos for apples-to-apples means.
