# FlashVSR RoPE Construction Site — Inspection Note (Task 4)

**Date:** 2026-07-02
**Repo inspected:** `github.com/OpenImagingLab/FlashVSR` @ `main` (shallow clone 2026-07-02)
**Purpose:** pin the exact temporal-RoPE injection point for the extrapolation probe
(`docs/superpowers/plans/2026-07-02-rope-extrapolation-vsr.md`, Task 5 consumes this note).

## Summary

FlashVSR is a DiffSynth-Studio fork. The Wan2.1 DiT uses **3D RoPE** with per-axis
precomputed complex frequency tables (temporal, H, W). Temporal positions enter as a
**slice index into a precomputed table** — not a `torch.arange` inside the attention
call — which makes the position override trivial: replace the slice with fancy
indexing by our `temporal_indices()` list.

## The RoPE code (all in `diffsynth/models/wan_video_dit.py`)

| what | where | detail |
|---|---|---|
| Table builder (per axis) | `wan_video_dit.py:257-262` `precompute_freqs_cis(dim, end=1024, theta=10000.0)` | `freqs = 1/(θ^(2k/d))`; `torch.outer(torch.arange(end), freqs)`; `torch.polar(...)` → complex tensor of shape `(end=1024, dim/2)`. **Positions are the row index 0..1023.** |
| 3D wrapper | `wan_video_dit.py:250-254` `precompute_freqs_cis_3d(dim, end=1024, theta=10000.0)` | temporal gets `dim - 2*(dim//3)` dims, H and W get `dim//3` each. Returns `(f_freqs, h_freqs, w_freqs)`. |
| Table instantiation | `wan_video_dit.py:563` (`WanModel.__init__`) | `self.freqs = precompute_freqs_cis_3d(head_dim)` with `head_dim = dim // num_heads`. |
| Application to q/k | `wan_video_dit.py:265-269` `rope_apply(x, freqs, num_heads)` | view-as-complex multiply; called at `:330-331` inside self-attention. |
| Non-streaming forward | `wan_video_dit.py:645-650` (`WanModel.forward`) | `freqs = cat(self.freqs[0][:f]···, self.freqs[1][:h]···, self.freqs[2][:w]···)` — temporal positions `0..f-1`. |

## The streaming pipelines build RoPE themselves (the actual injection point)

Each FlashVSR pipeline has its own `model_fn_wan_video(...)` that **bypasses
`WanModel.forward`** and constructs `freqs` before calling the blocks directly:

| pipeline | RoPE construction |
|---|---|
| `diffsynth/pipelines/flashvsr_tiny.py` | `:520-532` (inside `model_fn_wan_video`, def at `:489`) |
| `diffsynth/pipelines/flashvsr_tiny_long.py` | `:521-532` |
| `diffsynth/pipelines/flashvsr_full.py` | `:537-548` |

The construction (identical in all three):

```python
if cur_process_idx == 0:
    ... dit.freqs[0][:f] ...                                  # positions 0..f-1 (f=6)
else:
    ... dit.freqs[0][4 + cur_process_idx*2 : 4 + cur_process_idx*2 + f] ...   # f=2
```

## How temporal positions relate to the video (the arithmetic)

- Wan causal VAE + patchify(1,2,2): pixel frames `T` → latent frames `(T-1)/4`
  (buffer mode, `flashvsr_tiny.py:333`); `h=H/16`, `w=W/16`.
- Chunk loop (`flashvsr_tiny.py:339,352,386`): `process_total_num = (num_frames-1)//8 - 2`;
  chunk 0 processes latent slice `[0:6]`; chunk `i>0` processes `latents[:, :, 4+2i : 6+2i]`.
- **The RoPE temporal index == the absolute latent index** (`4+2i .. 5+2i`).
  Positions grow linearly with stream progress and never reset.

Consequences for the probe:

1. **Extrapolation is built in:** a long video streamed through FlashVSR reaches
   temporal RoPE positions ≫ anything a short training clip contained. The default
   example (81 frames) only exercises positions 0..19. Position `p` is reached at
   pixel-frame ≈ `4p` — a 1-minute 30 fps video reaches position ~450.
2. **Hard ceiling at 1024:** the table has 1024 rows (`end=1024`). At latent position
   1024 (≈ 4097 pixel frames ≈ 2.3 min @ 30 fps) the slice `dit.freqs[0][start:start+f]`
   silently returns fewer than `f` rows → shape mismatch crash in `rope_apply`.
   Extrapolation experiments beyond 1023 must rebuild the table with a larger `end`
   via `precompute_freqs_cis(dim - 2*(dim//3), end=NEW_END)` (pure function, safe).
3. **Trained temporal range:** not stated in the repo config; training used
   `num_frames=81`-style clips (the pipelines default to 81). To confirm from the
   paper, but positions ≳ 20 are plausibly already outside the densely-trained range.

## Injection design for Task 5 (what the hook must do)

Patch the **pipeline's** `model_fn_wan_video` (only the pipeline actually used —
start with `flashvsr_tiny.py`), replacing the two slice expressions with index-list
lookups:

```python
# baseline chunk-0 positions: list(range(f)) ; chunk-i: [4+2i, 5+2i]
t_idx = temporal_indices(f, override, base_start=<chunk start>)   # our Task-1 logic
freqs_t = dit.freqs[0][torch.tensor(t_idx)]                        # fancy index, same shape
```

Because the table lookup is the *only* place temporal position enters, a no-op
override (identical indices) is bit-identical by construction — the faithfulness
gate should pass trivially unless the hook is mis-wired.

Note for the shift/stretch semantics: the natural formulation on the streaming path
is **per-chunk**: baseline indices for chunk `i` are `[4+2i, 5+2i]` (or `range(6)` for
chunk 0); shift adds `k` to those, stretch multiplies them by `s`. `temporal_indices`
from Task 1 generates `[round(j*s)+k for j in base_indices]` — pass the chunk's base
indices via the explicit-`indices` field or extend the helper with a `base` argument
(Task 5 decision).

## Model / weights / env facts (for the baseline command)

- Weights: HF repo `JunhaoZhuang/FlashVSR-v1.1` → `examples/WanVSR/FlashVSR-v1.1/`
  (`diffusion_pytorch_model_streaming_dmd.safetensors`, `LQ_proj_in.ckpt`,
  `TCDecoder.ckpt`, VAE). Downloaded on the server via `HF_ENDPOINT=https://hf-mirror.com`.
- Entry points: `examples/WanVSR/infer_flashvsr_v1.1_tiny.py` (short),
  `infer_flashvsr_v1.1_tiny_long_video.py` (long/streaming). Input: dir of PNG/JPG
  frames (natural-sorted) or a video file; frames padded to `8n+1`; output dims
  forced to multiples of 128; default `scale=4.0`, one denoising step
  (`num_inference_steps=1`), `seed=0`.
- Env: conda `flashvsr`, Python 3.11, torch 2.6.0+cu124 (server has CUDA 12.4
  toolkit + driver 550.144.03, A100 = officially supported for Block-Sparse-Attention).
- Server setup script: `/tmp/flashvsr_setup.sh` (staged, resumable), log at
  `/tmp/flashvsr_setup.log`, tmux session `flashvsr_setup`.

## Baseline run + VRAM (Task 4 Step 2 gate) — PASSED 2026-07-02

Smoke test: stock v1.1 tiny pipeline on the bundled `example0.mp4`
(384×384, 85 frames → ×4 → 1536×1536, 89 padded frames, 9 streaming chunks),
driven by an external wrapper (`/tmp/flashvsr_smoke.py` on the server) that
imports the stock `infer_flashvsr_v1.1_tiny.py` by file path — **zero repo
modification**.

- **Command** (tmux, GPU 0):
  `cd ~/repos/FlashVSR/examples/WanVSR && CUDA_VISIBLE_DEVICES=0 python /tmp/flashvsr_smoke.py`
  (wrapper must `sys.path.insert(0, os.getcwd())` — the stock script imports
  `utils.*` relative to `examples/WanVSR`; weights paths are cwd-relative too)
- **Result:** 24.2 s inference, **peak VRAM 24.66 GiB** at 1536×1536 output
  (fits the 40 GB A100 with headroom; our probe outputs, 1280×720, will be far
  smaller), 3.67 fps under heavy tenant contention.
- **Output sanity:** 85 PNGs at `~/results/rope_probe/_baseline_smoke/frames/`;
  stats healthy (mean ≈ 118.5, std ≈ 60, frame0→1 mean-abs-diff 2.47 — real
  content, temporally continuous).

## Pristine-repo guarantee (for LR-VCC benchmark reuse)

FlashVSR may later be evaluated *unmodified* as a method row in the LR-VCC
benchmark, so the probe never edits the repo:

- Server clone pinned at commit `b527c6f`, tagged **`pristine-2026-07-02`**;
  `git status` clean (only the untracked `FlashVSR-v1.1/` weights dir).
- The probe hook (Task 5) is a **runtime monkeypatch** in our own
  `scripts/rope_probe/flashvsr_hook.py`; verify before any run with
  `cd ~/repos/FlashVSR && git status --short`.
- diffsynth is importable via a `.pth` file in the `flashvsr` env's
  site-packages pointing at `~/repos/FlashVSR` (the editable install fails on
  the setuptools-81 `pkg_resources` gotcha; `.pth` is equivalent and clean).

## Task 5 gate — PASSED 2026-07-02 (bit-exact)

Hook design (better than planned): instead of monkeypatching a pipeline
function, `scripts/rope_probe/flashvsr_hook.py` swaps `dit.freqs[0]` for a
duck-typed `TemporalFreqTable` whose `__getitem__` routes every slice through
the `PositionOverride` — one injection point covers tiny / tiny_long / full /
`WanModel.forward`; no-op returns the original table view (bit-exact by
construction); out-of-range positions served from an on-demand extended table
built by the model's own `precompute_freqs_cis` (prefix bitwise-asserted).

`verify_noop.py` five-run gate on hhsz first 85 frames (GPU 0, one process,
seed 0), comparing raw bf16 outputs upcast to fp32:

| check | value | meaning |
|---|---|---|
| floor (unhooked rerun) | **0.0** | pipeline fully deterministic in-process |
| no-op drift | **0.0** | hooked no-op bit-exact — gate PASS |
| shift=+1 diff | 0.2949 | hook engages (and: +1 latent position visibly changes output) |
| after restore() | **0.0** | restore clean |

The shift-engagement check exists to kill the silent-no-wire failure mode; its
magnitude is *not* yet evidence about H0 (could be bf16 phase-rounding
amplification, first-chunk boundary, or genuine leakage — Phase 1's sweep with
per-frame aggregate metrics answers that properly).

## Task 6 first sweeps — hhsz 85 frames, pyiqa-scored (2026-07-03)

Self-consistency vs the in-process baseline; **pyiqa** (IQA-PyTorch), PSNR/SSIM
RGB (`test_y_channel=False`, DOVE convention), pyiqa LPIPS — NOT
skimage/lpips-pkg, so numbers are comparable with the project's baselines.
Driver saves frames only; `score_conditions.py` scores separately (vsr env).
JSONs: `results/rope_probe/{shift,stretch}/hhsz85/`.

**Shift control (H0)** — k ∈ {2, 8, 32, 128, 512, 996}, positions stay in-table:

PSNR 52.9–53.1 dB, SSIM 0.9976–0.9978, LPIPS 0.0006 — **flat, no trend in k**
even at the table edge (k=996 → top position 1020). The sparse streaming
attention is effectively shift-invariant in-range: RoPE's relative property
survives LCSA + bf16; no meaningful absolute-position leakage. (H0: no
leakage — the 0.295 max-abs from the gate's shift-1 check was numerics-level
at PSNR ≈ 53 dB, as suspected.)

**Stretch (toward H1)** — s ∈ {2…64}, same frames, positions spread out:

| s | PSNR | SSIM | LPIPS |
|---|-----:|-----:|------:|
| 2 | 35.65 | 0.9460 | 0.0419 |
| 4 | 32.72 | 0.9125 | 0.0744 |
| 8 | 32.14 | 0.9019 | 0.0825 |
| 16 | 32.64 | 0.9084 | 0.0788 |
| 32 | 33.58 | 0.9230 | 0.0651 |
| 64 | 32.10 | 0.8988 | 0.0877 |

Reads: (1) stretching relative distances changes output ~20 dB more than any
in-range shift — the model is sensitive to position *geometry*, not position
*offset*; (2) the response saturates after s≈4–8 with a mild non-monotonic
wobble (s=16/32 slightly closer to baseline than s=8 — possibly RoPE
frequency periodicity; worth a finer sweep); (3) s=64 pushes positions to
~1536, past the 1024-row table — the extended-table path ran in production
and output degraded smoothly, **no crash / no collapse** on this 85-frame
clip.

Caveats: self-consistency measures *change*, not *quality* — the H1 verdict
needs vs-GT and LR-VCC on longer clips (next: longer-window sweeps + Task 7
LR-VCC pass + Task 8 curves). A correctly-relative attention is *expected* to
respond to stretch (geometry changes); the question is whether output
*quality* degrades where long videos actually operate.

## The "21-frame window" (supervisor discussion 2026-07-03) — code-level mapping

Group discussion framed the goal as: the model has a **trained window of 21
(latent) / ~81 (pixel) frames**; beyond it RoPE must extrapolate; another
student will *extend* the window and check quality. Code confirms all of it:

- 81 pixel frames = pipeline default `num_frames=81` (`flashvsr_tiny.py:291`);
  causal-VAE 4× time compression → (81−1)/4 + 1 = **21 latent frames** — the
  DiT's trained temporal window; RoPE positions 0..20 are the trained rows.
- **Streaming avoids distance extrapolation by construction:** attention span
  per chunk = 2 current latents + `kv_len=3` cached windows (cache trim in
  `wan_video_dit.py:368-375` region) ≈ 8 latents ≈ ~30 pixel frames. Relative
  distances stay far inside the trained window on any-length stream; only the
  **absolute** position magnitude grows (`4+2i`, unbounded).
- Our sweeps decompose exactly these two factors:
  **magnitude beyond trained window (shift) = free** (53 dB flat, clip parked
  at 996–1020 ≫ 20); **distance beyond trained window (stretch) = what
  bites** (~32 dB, saturating).

Implication for the window-extension work (the other student): a window of N
latents has relative distances up to N — at N > 21 that is distance
extrapolation, our "stretch" axis, not our benign "shift" axis. Predicted:
moderate, saturating degradation. **Causal tool we can hand over:** run the
extended window twice, stock positions vs `stretch = 21/N` (position
*compression* — LLM-style Position Interpolation via our hook, no repo edit).
Quality recovers → RoPE extrapolation is the bottleneck (and PI is the fix);
doesn't recover → capacity/content, not positions. A fractional+fine stretch
sweep (s ∈ {0.25…3.0}) is running to chart the PI side of that curve.

## Server env facts (standup 2026-07-02, all resolved)

- conda env **`flashvsr`** (py3.11): torch 2.6.0+cu124, FlashVSR requirements
  (torch lines excluded), `modelscope` (diffsynth hard-imports it).
- **Block-Sparse-Attention** compiled from source with
  `BLOCK_SPARSE_ATTN_CUDA_ARCHS="80"` — upstream setup.py emits a
  `compute_120` gencode that CUDA 12.4's nvcc rejects (`nvcc fatal`); pinning
  the arch list to sm_80 (A100) fixes it. ~15 min at `MAX_JOBS=8`.
- Import-order gotcha: `import torch` **before** `block_sparse_attn`
  (the CUDA extension needs torch's `libc10.so` on the loader path).
- Setup scripts (staged/resumable, logs in `/tmp`): `/tmp/flashvsr_setup.sh`,
  `/tmp/flashvsr_fix45.sh`; stage flags `/tmp/flashvsr_setup.stage[1-6].done`.
